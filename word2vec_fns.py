import tensorflow as tf
import numpy as np
import collections

data_index = 0

def generate_batch(data, batch_size, skip_window):
    """
    Generates a mini-batch of training data for the training CBOW
    embedding model.
    :param data (numpy.ndarray(dtype=int, shape=(corpus_size,)): holds the
        training corpus, with words encoded as an integer
    :param batch_size (int): size of the batch to generate
    :param skip_window (int): number of words to both left and right that form
        the context window for the target word.
    Batch is a vector of shape (batch_size, 2*skip_window), with each entry for the batch containing all the context words, with the corresponding label being the word in the middle of the context
    """
    global data_index
        
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    batch = np.ndarray(shape=(batch_size, span-1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
   
    buffer = collections.deque(maxlen=span)
    #collect the first window of words
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)   
    #move the sliding window
    for i in range(batch_size):
        target = skip_window
        col_idx = 0
        for j in range(span):
            if j==span//2:
                continue
            batch[i,col_idx] = buffer[j]
            col_idx += 1
        labels[i, 0] = buffer[target] #the word at the center
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    assert batch.shape[0] == batch_size and batch.shape[1] == span - 1
    return batch, labels

def get_mean_context_embeds(embeddings, train_inputs):
    """
    :param embeddings (tf.Variable(shape=(vocabulary_size, embedding_size))
    :param train_inputs (tf.placeholder(shape=(batch_size, 2*skip_window))
    returns:
        `mean_context_embeds`: the mean of the embeddings for all context words
        for each entry in the batch, should have shape (batch_size,
        embedding_size)
    """
    # cpu is recommended to avoid out of memory errors, if you don't
    # have a high capacity GPU
    with tf.device('/cpu:0'):
        embeds = None
        
        for i in range(2*skip_window):
            
            embedding_i = tf.nn.embedding_lookup(embeddings, train_inputs[:,i])
            
            emb_x,emb_y = embedding_i.get_shape().as_list()
            
            if embeds is None:
                
                embeds = tf.reshape(embedding_i,[emb_x,emb_y,1])
                
            else:
                
                embeds = tf.concat(2,[embeds,tf.reshape(embedding_i,[emb_x,emb_y,1])])
                
    assert embeds.get_shape().as_list()[2]==2*skip_window
    
    mean_context_embeds =  tf.reduce_mean(embeds,2,keep_dims=False)
	    
    return mean_context_embeds
