from nets import vgg
import tensorflow as tf
from preprocessing import vgg_preprocessing
from upsampling import bilinear_upsample_weights

slim = tf.contrib.slim

# Mean values for VGG-16
from preprocessing.vgg_preprocessing import _R_MEAN, _G_MEAN, _B_MEAN

wd = 5e-4

def leaky_relu(x, alpha=0.01):

    return tf.maximum(x, alpha * x)

def _score_layer(wd, bottom, in_features, kernel_size, name, num_classes):

    with tf.variable_scope(name) as scope:
        # get number of input channels
        shape = [kernel_size, kernel_size, in_features, num_classes]
        # He initialization Sheme
        if name == "fc_9_1":
            stddev = 0.0001
       

        # Apply convolution
        w_decay = wd

        weights = _variable_with_weight_decay(shape, stddev, w_decay,
                                                    decoder=True)
        conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
        # Apply bias
        conv_biases = _bias_variable([num_classes], constant=0.0)
        bias = tf.nn.bias_add(conv, conv_biases)

        return bias


def _variable_with_weight_decay(shape, stddev, wd, decoder=False):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.
        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.
        Returns:
          Variable Tensor
        """

        #initializer = tf.contrib.layers.variance_scaling_initializer()
        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        
        return var


def _bias_variable(shape, constant=0.0):

        initializer = tf.constant_initializer(constant)
        var = tf.get_variable(name='biases', shape=shape,
                              initializer=initializer)
       

        return var

def FCN_8s(image_batch_tensor,
           number_of_classes,
           new_number_of_classes,
           is_training,
           is_reuse,
           is_fine_tune=False):
    """Returns the FCN-8s model definition.
    The function returns the model definition of a network that was described
    in 'Fully Convolutional Networks for Semantic Segmentation' by Long et al.
    The network subsamples the input by a factor of 32 and uses three bilinear
    upsampling layers to upsample prediction by a factor of 32. This means that
    if the image size is not of the factor 32, the prediction of different size
    will be delivered. To adapt the network for an any size input use 
    adapt_network_for_any_size_input(FCN_8s, 32). Note: the upsampling kernel
    is fixed in this model definition, because it didn't give significant
    improvements according to aforementioned paper.
    
    Parameters
    ----------
    image_batch_tensor : [batch_size, height, width, depth] Tensor
        Tensor specifying input image batch
    number_of_classes : int
        An argument specifying the number of classes to be predicted.
        For example, for PASCAL VOC it is 21.
    is_training : boolean
        An argument specifying if the network is being evaluated or trained.
        It affects the work of underlying dropout layer of VGG-16.
    
    Returns
    -------
    upsampled_logits : [batch_size, height, width, number_of_classes] Tensor
        Tensor with logits representing predictions for each class.
        Be careful, the output can be of different size compared to input,
        use adapt_network_for_any_size_input to adapt network for any input size.
        Otherwise, the input images sizes should be of multiple 32.
    fcn_16s_variables_mapping : dict {string: variable}
        Dict which maps the FCN-8s model's variables to FCN-16s checkpoint variables
        names. We need this to initilize the weights of FCN-8s model with FCN-16s from
        checkpoint file. Look at ipython notebook for examples.
    """

    # Convert image to float32 before subtracting the
    # mean pixel value
    image_batch_float = tf.to_float(image_batch_tensor)

    # Subtract the mean pixel value from each pixel
    mean_centered_image_batch = image_batch_float - [_R_MEAN, _G_MEAN, _B_MEAN]

    upsample_filter_factor_2_np = bilinear_upsample_weights(factor=2,
                                                            number_of_classes=number_of_classes)

    upsample_filter_factor_8_np = bilinear_upsample_weights(factor=8,
                                                             number_of_classes=number_of_classes)

    upsample_filter_factor_2_tensor = tf.constant(upsample_filter_factor_2_np)
    upsample_filter_factor_8_tensor = tf.constant(upsample_filter_factor_8_np)

    with tf.variable_scope('FCN_slice', reuse = is_reuse):
       
       with tf.variable_scope("fcn_8s")  as fcn_8s_scope:
            # Define the model that we want to use -- specify to use only two classes at the last layer
            # TODO: make pull request to get this custom vgg feature accepted
            # to avoid using custom slim repo.
            with slim.arg_scope(vgg.vgg_arg_scope()):

                ## Original FCN-32s model definition

                last_layer_logits, end_points = vgg.vgg_16(mean_centered_image_batch,
                                                        num_classes=number_of_classes,
                                                        is_training=is_training,
                                                        dropout_keep_prob=0.5,
                                                        spatial_squeeze=False,
                                                        use_dilated=False,
                                                        scope='vgg_16',
                                                        fc_conv_padding='SAME',
                                                        global_pool=False)


                last_layer_logits_shape = tf.shape(last_layer_logits)


                # Calculate the ouput size of the upsampled tensor
                last_layer_upsampled_by_factor_2_logits_shape = tf.stack([
                                                                    last_layer_logits_shape[0],
                                                                    last_layer_logits_shape[1] * 2,
                                                                    last_layer_logits_shape[2] * 2,
                                                                    last_layer_logits_shape[3]
                                                                    ])

                
                last_layer_logits = slim.batch_norm(last_layer_logits, activation_fn=tf.nn.relu)
                
                # Perform the upsampling
                last_layer_upsampled_by_factor_2_logits = tf.nn.conv2d_transpose(last_layer_logits,
                                                                                upsample_filter_factor_2_tensor,
                                                                                output_shape=last_layer_upsampled_by_factor_2_logits_shape,
                                                                                strides=[1, 2, 2, 1],
                                                                                name='upscore2')

                ## Adding the skip here for FCN-16s model
                
                # We created vgg in the fcn_8s name scope -- so
                # all the vgg endpoints now are prepended with fcn_8s name
            
                pool4_features = end_points['FCN_slice/fcn_8s/vgg_16/pool4']

                # We zero initialize the weights to start training with the same
                # accuracy that we ended training FCN-32s

                pool4_features = slim.batch_norm(pool4_features, activation_fn=tf.nn.relu)
                
                pool4_logits = slim.conv2d(pool4_features,
                                        number_of_classes,
                                        [1, 1],
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        weights_initializer=tf.zeros_initializer,
                                        scope='pool4_fc')


                
                fused_last_layer_and_pool4_logits = pool4_logits + last_layer_upsampled_by_factor_2_logits

                fused_last_layer_and_pool4_logits_shape = tf.shape(fused_last_layer_and_pool4_logits)
                
                
                

                # Calculate the ouput size of the upsampled tensor
                fused_last_layer_and_pool4_upsampled_by_factor_2_logits_shape = tf.stack([
                                                                            fused_last_layer_and_pool4_logits_shape[0],
                                                                            fused_last_layer_and_pool4_logits_shape[1] * 2,
                                                                            fused_last_layer_and_pool4_logits_shape[2] * 2,
                                                                            fused_last_layer_and_pool4_logits_shape[3]
                                                                            ])

                fused_last_layer_and_pool4_logits = slim.batch_norm(fused_last_layer_and_pool4_logits, activation_fn=tf.nn.relu)
                
                # Perform the upsampling
                fused_last_layer_and_pool4_upsampled_by_factor_2_logits = tf.nn.conv2d_transpose(fused_last_layer_and_pool4_logits,
                                                                            upsample_filter_factor_2_tensor,
                                                                            output_shape=fused_last_layer_and_pool4_upsampled_by_factor_2_logits_shape,
                                                                            strides=[1, 2, 2, 1],
                                                                            name='upscore4')
                
                
                ## Adding the skip here for FCN-8s model

                pool3_features = end_points['FCN_slice/fcn_8s/vgg_16/pool3']
                
                # We zero initialize the weights to start training with the same
                # accuracy that we ended training FCN-32s

                pool3_features = slim.batch_norm(pool3_features, activation_fn=tf.nn.relu)
                
                pool3_logits = slim.conv2d(pool3_features,
                                        number_of_classes,
                                        [1, 1],
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        weights_initializer=tf.zeros_initializer,
                                        scope='pool3_fc')

                
                fused_last_layer_and_pool4_logits_and_pool_3_logits = pool3_logits + \
                                                fused_last_layer_and_pool4_upsampled_by_factor_2_logits
                
                
                fused_last_layer_and_pool4_logits_and_pool_3_logits_shape = tf.shape(fused_last_layer_and_pool4_logits_and_pool_3_logits)
                
                
                # Calculate the ouput size of the upsampled tensor
                fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits_shape = tf.stack([
                                                                            fused_last_layer_and_pool4_logits_and_pool_3_logits_shape[0],
                                                                            fused_last_layer_and_pool4_logits_and_pool_3_logits_shape[1] * 8,
                                                                            fused_last_layer_and_pool4_logits_and_pool_3_logits_shape[2] * 8,
                                                                            fused_last_layer_and_pool4_logits_and_pool_3_logits_shape[3]
                                                                            ])

                fused_last_layer_and_pool4_logits_and_pool_3_logits = slim.batch_norm(fused_last_layer_and_pool4_logits_and_pool_3_logits, activation_fn=tf.nn.relu)
                
                # Perform the upsampling
                fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits = tf.nn.conv2d_transpose(fused_last_layer_and_pool4_logits_and_pool_3_logits,
                                                                            upsample_filter_factor_8_tensor,
                                                                            output_shape=fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits_shape,
                                                                            strides=[1, 8, 8, 1],
                                                                            name='upscore32')
                
                
                fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits = slim.batch_norm(fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits, activation_fn=tf.nn.relu)


                fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits_3 = \
                _score_layer(wd, fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits, number_of_classes, 1, 'fc_9_1', new_number_of_classes)


                fcn_8s_variables_mapping = {}

                fcn_8s_variables = slim.get_variables(fcn_8s_scope)
                
                for variable in fcn_8s_variables:
                    
                    # We only need FCN-16s variables to resture from checkpoint
                    # Variables of FCN-8s should be initialized
                    if not is_fine_tune:
                        if 'fc_9_1' in variable.name:
                            continue
                    
                    # Here we remove the part of a name of the variable
                    # that is responsible for the current variable scope
                    original_fcn_8s_checkpoint_string = 'FCN_slice/fcn_8s' +  variable.name[len(fcn_8s_scope.name):-2]
                    fcn_8s_variables_mapping[original_fcn_8s_checkpoint_string] = variable


    return fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits_3, fcn_8s_variables_mapping