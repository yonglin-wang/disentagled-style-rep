import json
import logging
import numpy as np
import os
import pickle
import tensorflow as tf

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.config.model_config import mconf
from linguistic_style_transfer_model.evaluators import content_preservation, style_transfer
from linguistic_style_transfer_model.models.adversarial_autoencoder import AdversarialAutoencoder
from linguistic_style_transfer_model.utils import data_processor

logger = logging.getLogger(global_config.logger_name)


class AdversarialDeterministicAutoencoder(AdversarialAutoencoder):

    def __init__(self):
        super().__init__()
        logger.info("Building Deterministic Autoencoder")

    def get_style_embedding(self, sentence_embedding):

        style_embedding = tf.nn.dropout(
            x=tf.layers.dense(
                inputs=sentence_embedding, units=mconf.style_embedding_size,
                activation=tf.nn.leaky_relu, name="style_embedding"),
            keep_prob=mconf.fully_connected_keep_prob)

        return style_embedding

    def get_content_embedding(self, sentence_embedding):

        content_embedding = tf.nn.dropout(
            x=tf.layers.dense(
                inputs=sentence_embedding, units=mconf.content_embedding_size,
                activation=tf.nn.leaky_relu, name="content_embedding"),
            keep_prob=mconf.fully_connected_keep_prob)

        return content_embedding

    def build_model(self, word_index, encoder_embedding_matrix, decoder_embedding_matrix, num_labels):

        # model inputs
        self.input_sequence = tf.placeholder(
            dtype=tf.int32, shape=[None, global_config.max_sequence_length],
            name="input_sequence")
        logger.debug("input_sequence: {}".format(self.input_sequence))

        batch_size = tf.shape(self.input_sequence)[0]
        logger.debug("batch_size: {}".format(batch_size))

        self.input_label = tf.placeholder(
            dtype=tf.float32, shape=[None, num_labels], name="input_label")
        logger.debug("input_label: {}".format(self.input_label))

        self.sequence_lengths = tf.placeholder(
            dtype=tf.int32, shape=[None], name="sequence_lengths")
        logger.debug("sequence_lengths: {}".format(self.sequence_lengths))

        self.input_bow_representations = tf.placeholder(
            dtype=tf.float32, shape=[None, global_config.vocab_size],
            name="input_bow_representations")
        logger.debug("input_bow_representations: {}".format(self.input_bow_representations))

        self.conditioned_generation_mode = tf.placeholder(dtype=tf.bool, name="conditioned_generation_mode")
        logger.debug("conditioned_generation_mode: {}".format(self.conditioned_generation_mode))

        self.conditioning_embedding = tf.placeholder(
            dtype=tf.float32, shape=[None, mconf.style_embedding_size],
            name="conditioning_embedding")
        logger.debug("conditioning_embedding: {}".format(self.conditioning_embedding))

        self.epoch = tf.placeholder(dtype=tf.float32, shape=(), name="epoch")
        logger.debug("epoch: {}".format(self.epoch))

        decoder_input = tf.concat(
            values=[tf.fill(dims=[batch_size, 1], value=word_index[global_config.sos_token]),
                    self.input_sequence], axis=1, name="decoder_input")

        with tf.device('/cpu:0'):
            with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
                # word embeddings matrices
                encoder_embeddings = tf.get_variable(
                    initializer=encoder_embedding_matrix, dtype=tf.float32,
                    trainable=True, name="encoder_embeddings")
                logger.debug("encoder_embeddings: {}".format(encoder_embeddings))

                decoder_embeddings = tf.get_variable(
                    initializer=decoder_embedding_matrix, dtype=tf.float32,
                    trainable=True, name="decoder_embeddings")
                logger.debug("decoder_embeddings: {}".format(decoder_embeddings))

                # embedded sequences
                encoder_embedded_sequence = tf.nn.dropout(
                    x=tf.nn.embedding_lookup(params=encoder_embeddings, ids=self.input_sequence),
                    keep_prob=mconf.sequence_word_keep_prob,
                    name="encoder_embedded_sequence")
                logger.debug("encoder_embedded_sequence: {}".format(encoder_embedded_sequence))

                decoder_embedded_sequence = tf.nn.dropout(
                    x=tf.nn.embedding_lookup(params=decoder_embeddings, ids=decoder_input),
                    keep_prob=mconf.sequence_word_keep_prob,
                    name="decoder_embedded_sequence")
                logger.debug("decoder_embedded_sequence: {}".format(decoder_embedded_sequence))

        sentence_embedding = self.get_sentence_embedding(encoder_embedded_sequence)

        style_keep_probabilities = tf.random_uniform(shape=[batch_size], maxval=1, dtype=tf.float32)
        style_filter = style_keep_probabilities < mconf.style_embedding_keep_prob
        logger.debug("style_filter: {}".format(style_filter))

        # style embedding
        self.style_embedding = self.get_style_embedding(sentence_embedding)
        filtered_style_embedding = tf.where(
            condition=style_filter, x=self.style_embedding,
            y=tf.zeros_like(self.style_embedding))

        final_style_embedding = tf.cond(
            pred=self.conditioned_generation_mode,
            true_fn=lambda: self.conditioning_embedding,
            false_fn=lambda: self.style_embedding)
        logger.debug("style_embedding: {}".format(final_style_embedding))

        # content embedding
        self.content_embedding = self.get_content_embedding(sentence_embedding)
        logger.debug("content_embedding: {}".format(self.content_embedding))

        # concatenated generative embedding
        generative_embedding = tf.layers.dense(
            inputs=tf.concat(values=[final_style_embedding, self.content_embedding], axis=1),
            units=mconf.decoder_rnn_size, activation=tf.nn.leaky_relu,
            name="generative_embedding")
        logger.debug("generative_embedding: {}".format(generative_embedding))

        # sequence predictions
        with tf.name_scope('sequence_prediction'):
            training_output, self.inference_output, self.final_sequence_lengths = \
                self.generate_output_sequence(
                    decoder_embedded_sequence, generative_embedding, decoder_embeddings,
                    word_index, batch_size)
            logger.debug("training_output: {}".format(training_output))
            logger.debug("inference_output: {}".format(self.inference_output))

        # adversarial loss
        with tf.name_scope('adversarial_loss'):
            adversarial_label_prediction = self.get_adversarial_label_prediction(
                self.content_embedding, num_labels)
            logger.debug("adversarial_label_prediction: {}".format(adversarial_label_prediction))

            self.adversarial_label_prediction_hardmax = tf.contrib.seq2seq.hardmax(
                logits=adversarial_label_prediction, name="adversarial_label_prediction_hardmax")

            self.adversarial_entropy = \
                mconf.adversarial_discriminator_loss_weight * \
                tf.reduce_mean(
                    input_tensor=tf.reduce_sum(
                        input_tensor=-adversarial_label_prediction *
                                     tf.log(adversarial_label_prediction + mconf.epsilon), axis=1))
            logger.debug("adversarial_entropy: {}".format(self.adversarial_entropy))

            self.adversarial_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.input_label, logits=adversarial_label_prediction, label_smoothing=0.1)
            logger.debug("adversarial_loss: {}".format(self.adversarial_loss))

            bow_prediction = self.get_bow_prediction(self.style_embedding)
            self.bow_prediction_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.input_bow_representations, logits=bow_prediction, label_smoothing=0.1)
            logger.debug("bow_prediction_loss: {}".format(self.bow_prediction_loss))

            self.bow_entropy = \
                mconf.adversarial_bow_loss_weight * \
                tf.reduce_mean(
                    input_tensor=tf.reduce_sum(
                        input_tensor=-bow_prediction * tf.log(bow_prediction + mconf.epsilon), axis=1))
            logger.debug("bow_entropy: {}".format(self.bow_entropy))

        # style prediction loss
        with tf.name_scope('style_prediction_loss'):
            style_label_prediction = \
                self.get_style_label_prediction(self.style_embedding, num_labels)
            logger.debug("style_label_prediction: {}".format(style_label_prediction))

            self.style_label_prediction_hardmax = tf.contrib.seq2seq.hardmax(
                logits=style_label_prediction, name="style_label_prediction_hardmax")

            self.style_prediction_loss = \
                mconf.style_prediction_loss_weight * \
                tf.losses.softmax_cross_entropy(
                    onehot_labels=self.input_label, logits=style_label_prediction, label_smoothing=0.1)
            logger.debug("style_prediction_loss: {}".format(self.style_prediction_loss))

        with tf.name_scope('overall_prediction_loss'):
            overall_label_prediction = tf.layers.dense(
                inputs=tf.concat(values=[self.style_embedding, self.content_embedding], axis=1),
                units=num_labels, activation=tf.nn.softmax,
                name="overall_label_prediction")
            logger.debug("overall_label_prediction: {}".format(overall_label_prediction))

            self.overall_label_prediction_hardmax = tf.contrib.seq2seq.hardmax(
                logits=overall_label_prediction, name="overall_label_prediction_hardmax")

            self.overall_prediction_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.input_label, logits=overall_label_prediction, label_smoothing=0.1)
            logger.debug("overall_prediction_loss: {}".format(self.overall_prediction_loss))

        # reconstruction loss
        with tf.name_scope('reconstruction_loss'):
            batch_maxlen = tf.reduce_max(self.sequence_lengths)
            logger.debug("batch_maxlen: {}".format(batch_maxlen))

            # the training decoder only emits outputs equal in time-steps to the
            # max time in the current batch
            target_sequence = tf.slice(
                input_=self.input_sequence,
                begin=[0, 0],
                size=[batch_size, batch_maxlen],
                name="target_sequence")
            logger.debug("target_sequence: {}".format(target_sequence))

            output_sequence_mask = tf.sequence_mask(
                lengths=tf.add(x=self.sequence_lengths, y=1),
                maxlen=batch_maxlen,
                dtype=tf.float32)

            self.reconstruction_loss = tf.contrib.seq2seq.sequence_loss(
                logits=training_output, targets=target_sequence,
                weights=output_sequence_mask)
            logger.debug("reconstruction_loss: {}".format(self.reconstruction_loss))

        # tensorboard logging variable summaries
        tf.summary.scalar(tensor=self.reconstruction_loss, name="reconstruction_loss_summary")
        tf.summary.scalar(tensor=self.style_prediction_loss, name="style_prediction_loss_summary")
        tf.summary.scalar(tensor=self.adversarial_loss, name="adversarial_loss_summary")

    def get_batch_indices(self, batch_number, data_limit):

        start_index = batch_number * mconf.batch_size
        end_index = min((batch_number + 1) * mconf.batch_size, data_limit)

        return start_index, end_index

    def run_batch(self, sess, start_index, end_index, fetches, padded_sequences,
                  one_hot_labels, text_sequence_lengths,
                  conditioning_embedding, conditioned_generation_mode, current_epoch):

        if not conditioned_generation_mode:
            conditioning_embedding = np.random.uniform(
                size=(end_index - start_index, mconf.style_embedding_size),
                low=-0.05, high=0.05).astype(dtype=np.float32)

        bow_representations = data_processor.get_bow_representations(
            padded_sequences[start_index: end_index])

        ops = sess.run(
            fetches=fetches,
            feed_dict={
                self.input_sequence: padded_sequences[start_index: end_index],
                self.input_label: one_hot_labels[start_index: end_index],
                self.sequence_lengths: text_sequence_lengths[start_index: end_index],
                self.input_bow_representations: bow_representations,
                self.conditioned_generation_mode: conditioned_generation_mode,
                self.conditioning_embedding: conditioning_embedding,
                self.epoch: current_epoch
            })

        return ops

    def train(self, sess, data_size, padded_sequences, text_sequence_lengths, one_hot_labels, num_labels,
              word_index, encoder_embedding_matrix, decoder_embedding_matrix, validation_sequences,
              validation_sequence_lengths, validation_labels, inverse_word_index, validation_actual_word_lists,
              options):

        writer = tf.summary.FileWriter(
            logdir="/tmp/tensorflow_logs/{}".format(global_config.experiment_timestamp),
            graph=sess.graph)

        trainable_variables = tf.trainable_variables()
        logger.debug("trainable_variables: {}".format(trainable_variables))

        self.composite_loss = 0.0
        self.composite_loss += self.reconstruction_loss
        self.composite_loss -= self.adversarial_entropy
        self.composite_loss += self.style_prediction_loss
        tf.summary.scalar(tensor=self.composite_loss, name="composite_loss")
        self.all_summaries = tf.summary.merge_all()

        adversarial_variable_labels = ["adversarial"]
        overall_classification_labels = ["overall_label_prediction"]

        # optimize adversarial classification
        adversarial_training_optimizer = tf.train.RMSPropOptimizer(
            learning_rate=mconf.adversarial_discriminator_learning_rate)
        adversarial_training_variables = [
            x for x in trainable_variables if any(
                scope in x.name for scope in adversarial_variable_labels)]
        logger.debug("adversarial_training_optimizer.variables: {}".format(adversarial_training_variables))
        adversarial_training_operation = None
        for i in range(mconf.adversarial_discriminator_iterations):
            adversarial_training_operation = adversarial_training_optimizer.minimize(
                loss=self.adversarial_loss,
                var_list=adversarial_training_variables)

        # optimize overall latent space classification
        overall_classification_optimizer = tf.train.AdamOptimizer(
            learning_rate=mconf.autoencoder_learning_rate)
        overall_classification_training_variables = [
            x for x in trainable_variables if any(
                scope in x.name for scope in overall_classification_labels)]
        logger.debug("overall_classification_training_variables: {}".format(
            overall_classification_training_variables))
        overall_classification_training_operation = overall_classification_optimizer.minimize(
            loss=self.overall_prediction_loss,
            var_list=overall_classification_training_variables)

        # optimize reconstruction
        reconstruction_training_optimizer = tf.train.AdamOptimizer(
            learning_rate=mconf.autoencoder_learning_rate)
        reconstruction_training_variables = [
            x for x in trainable_variables if all(
                scope not in x.name for scope in adversarial_variable_labels)]
        logger.debug("reconstruction_training_optimizer.variables: {}".format(reconstruction_training_variables))
        reconstruction_training_operation = None
        for i in range(mconf.autoencoder_iterations):
            reconstruction_training_operation = reconstruction_training_optimizer.minimize(
                loss=self.composite_loss, var_list=reconstruction_training_variables)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        num_batches = data_size // mconf.batch_size
        if data_size % mconf.batch_size:
            num_batches += 1
        logger.debug("Training - texts shape: {}; labels shape {}"
                     .format(padded_sequences.shape, one_hot_labels.shape))

        for current_epoch in range(1, options.training_epochs + 1):

            all_style_embeddings = list()
            all_content_embeddings = list()

            shuffle_indices = np.random.permutation(np.arange(data_size))

            shuffled_padded_sequences = padded_sequences[shuffle_indices]
            shuffled_one_hot_labels = one_hot_labels[shuffle_indices]
            shuffled_text_sequence_lengths = text_sequence_lengths[shuffle_indices]

            for batch_number in range(num_batches):
                (start_index, end_index) = self.get_batch_indices(
                    batch_number=batch_number, data_limit=data_size)

                logger.debug("start_index: {}, end_index: {}".format(start_index, end_index))

                fetches = \
                    [reconstruction_training_operation,
                     adversarial_training_operation,
                     overall_classification_training_operation,
                     self.reconstruction_loss,
                     self.style_prediction_loss,
                     self.adversarial_loss,
                     self.adversarial_entropy,
                     self.composite_loss,
                     self.style_embedding,
                     self.content_embedding,
                     self.all_summaries]

                [_, _, _,
                 reconstruction_loss,
                 style_loss,
                 adversarial_loss, adversarial_entropy,
                 composite_loss,
                 style_embeddings, content_embedding,
                 all_summaries] = \
                    self.run_batch(
                        sess, start_index, end_index, fetches,
                        shuffled_padded_sequences, shuffled_one_hot_labels,
                        shuffled_text_sequence_lengths, None, False, current_epoch)

                log_msg = "[R: {:.2f}, " \
                          "S: {:.2f}, " \
                          "ACE: {:.2f}, AE: {:.2f}] " \
                          "Epoch {}-{}: {:.4f} "
                logger.info(log_msg.format(
                    reconstruction_loss,
                    style_loss,
                    adversarial_loss, adversarial_entropy,
                    current_epoch, batch_number, composite_loss))

                all_style_embeddings.extend(style_embeddings)
                all_content_embeddings.extend(content_embedding)

                writer.add_summary(all_summaries)
                writer.flush()

            saver.save(sess=sess, save_path=global_config.model_save_path)

            with open(global_config.all_style_embeddings_path, 'wb') as pickle_file:
                pickle.dump(all_style_embeddings, pickle_file)
            with open(global_config.all_content_embeddings_path, 'wb') as pickle_file:
                pickle.dump(all_content_embeddings, pickle_file)
            with open(global_config.all_shuffled_labels_path, 'wb') as pickle_file:
                pickle.dump(shuffled_one_hot_labels, pickle_file)

            average_label_embeddings = data_processor.get_average_label_embeddings(
                data_size, options.dump_embeddings)

            with open(global_config.average_label_embeddings_path, 'wb') as pickle_file:
                pickle.dump(average_label_embeddings, pickle_file)

            if not current_epoch % global_config.validation_interval:

                logger.info("Running Validation {}:".format(current_epoch // global_config.validation_interval))

                glove_model = content_preservation.load_glove_model(options.validation_embeddings_file_path)

                validation_style_transfer_scores = list()
                validation_content_preservation_scores = list()
                validation_word_overlap_scores = list()
                for i in range(num_labels):

                    logger.info("validating label {}".format(i))

                    label_embeddings = list()
                    validation_sequences_to_transfer = list()
                    validation_labels_to_transfer = list()
                    validation_sequence_lengths_to_transfer = list()

                    for k in range(len(all_style_embeddings)):
                        if shuffled_one_hot_labels[k].tolist().index(1) == i:
                            label_embeddings.append(all_style_embeddings[k])

                    for k in range(len(validation_sequences)):
                        if validation_labels[k].tolist().index(1) != i:
                            validation_sequences_to_transfer.append(validation_sequences[k])
                            validation_labels_to_transfer.append(validation_labels[k])
                            validation_sequence_lengths_to_transfer.append(validation_sequence_lengths[k])

                    style_embedding = np.mean(np.asarray(label_embeddings), axis=0)

                    validation_batches = len(validation_sequences_to_transfer) // mconf.batch_size
                    if len(validation_sequences_to_transfer) % mconf.batch_size:
                        validation_batches += 1

                    validation_generated_sequences = list()
                    validation_generated_sequence_lengths = list()
                    for val_batch_number in range(validation_batches):
                        (start_index, end_index) = self.get_batch_indices(
                            batch_number=val_batch_number,
                            data_limit=len(validation_sequences_to_transfer))

                        conditioning_embedding = np.tile(
                            A=style_embedding, reps=(end_index - start_index, 1))

                        [validation_generated_sequences_batch, validation_sequence_lengths_batch] = \
                            self.run_batch(
                                sess, start_index, end_index,
                                [self.inference_output, self.final_sequence_lengths],
                                validation_sequences_to_transfer, validation_labels_to_transfer,
                                validation_sequence_lengths_to_transfer,
                                conditioning_embedding, True, current_epoch)
                        validation_generated_sequences.extend(validation_generated_sequences_batch)
                        validation_generated_sequence_lengths.extend(validation_sequence_lengths_batch)

                    trimmed_generated_sequences = \
                        [[index for index in sequence
                          if index != global_config.predefined_word_index[global_config.eos_token]]
                         for sequence in [x[:(y - 1)] for (x, y) in zip(
                            validation_generated_sequences, validation_generated_sequence_lengths)]]

                    generated_word_lists = \
                        [data_processor.generate_words_from_indices(x, inverse_word_index)
                         for x in trimmed_generated_sequences]

                    generated_sentences = [" ".join(x) for x in generated_word_lists]

                    output_file_path = "output/{}-training/validation_sentences_{}.txt".format(
                        global_config.experiment_timestamp, i)
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    with open(output_file_path, 'w') as output_file:
                        for sentence in generated_sentences:
                            output_file.write(sentence + "\n")

                    [style_transfer_score, confusion_matrix] = style_transfer.get_style_transfer_score(
                        options.classifier_saved_model_path, output_file_path, i)
                    logger.debug("style_transfer_score: {}".format(style_transfer_score))
                    logger.debug("confusion_matrix: {}".format(confusion_matrix))

                    content_preservation_score = content_preservation.get_content_preservation_score(
                        validation_actual_word_lists, generated_word_lists, glove_model)
                    logger.debug("content_preservation_score: {}".format(content_preservation_score))

                    word_overlap_score = content_preservation.get_word_overlap_score(
                        validation_actual_word_lists, generated_word_lists)
                    logger.debug("word_overlap_score: {}".format(word_overlap_score))

                    validation_style_transfer_scores.append(style_transfer_score)
                    validation_content_preservation_scores.append(content_preservation_score)
                    validation_word_overlap_scores.append(word_overlap_score)

                aggregate_style_transfer = np.mean(np.asarray(validation_style_transfer_scores))
                logger.info("Aggregate Style Transfer: {}".format(aggregate_style_transfer))

                aggregate_content_preservation = np.mean(np.asarray(validation_content_preservation_scores))
                logger.info("Aggregate Content Preservation: {}".format(aggregate_content_preservation))

                aggregate_word_overlap = np.mean(np.asarray(validation_word_overlap_scores))
                logger.info("Aggregate Word Overlap: {}".format(aggregate_word_overlap))

                with open(global_config.validation_scores_path, 'a+') as validation_scores_file:
                    validation_record = {
                        "epoch": current_epoch,
                        "style-transfer": aggregate_style_transfer,
                        "content-preservation": aggregate_content_preservation,
                        "word-overlap": aggregate_word_overlap
                    }
                    validation_scores_file.write(json.dumps(validation_record) + "\n")

        writer.close()