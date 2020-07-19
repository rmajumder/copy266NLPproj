
def assign_flag_values(tf):
    tf.app.flags.DEFINE_string('f', '', 'kernel')

    flags = tf.app.flags

    flags.DEFINE_string('ASC', 'qb', 'Aspect level sentiments from QB')
    flags.DEFINE_string('DSC', 'yelp', 'Document level sentiments from Yelp reviews')
    flags.DEFINE_integer('batch_size', 256, 'Training batch size')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
    flags.DEFINE_integer('n_iter', 35, 'training iteration')
    flags.DEFINE_float('gamma', 0.1, 'gamma')
    flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
    flags.DEFINE_integer('position_dim', 100, 'dimension of position embedding')
    flags.DEFINE_integer('max_sentence_len', 160, 'max number of tokens per sentence')
    flags.DEFINE_integer('max_target_len', 25, 'max number of tokens per target')
    flags.DEFINE_float('keep_prob1', 0.5, 'dropout keep prob1')
    flags.DEFINE_float('keep_prob2', 1.0, 'dropout keep prob2')
    flags.DEFINE_integer('filter_size', 3, 'filter_size')
    flags.DEFINE_integer('sc_num', 16, 'sc_num')
    flags.DEFINE_integer('sc_dim', 16, 'sc_dim')
    flags.DEFINE_integer('cc_num',  3, 'cc_num')
    flags.DEFINE_integer('cc_dim', 24, 'cc_dim')
    flags.DEFINE_integer('iter_routing', 3, 'routing iteration')
    flags.DEFINE_bool("reuse_embedding", False, "reuse word embedding & id, True or False")
    FLAGS = flags.FLAGS
    return FLAGS