# def facial_expression_model():
#     # first create the image model
#     images_model_input = Input(shape=(96, 96, 1))

#     # first conv layer
#     image_model_conv1 = Conv2D(64, (5, 5), activation='relu')(images_model_input)
#     image_model_max_pool1 = MaxPool2D(pool_size=(2, 2))(image_model_conv1)
#     image_model_batch_norm1 = BatchNormalization()(image_model_max_pool1)

#     # second conv layer
#     image_model_conv2 = Conv2D(64, (3, 3), activation='relu')(image_model_batch_norm1)
#     image_model_max_pool2 = MaxPool2D(pool_size=(2, 2))(image_model_conv2)
#     image_model_batch_norm2 = BatchNormalization()(image_model_max_pool2)

#     # third conv layer
#     image_model_conv3 = Conv2D(128, (3, 3), activation='relu')(image_model_batch_norm2)
#     image_model_max_pool3 = MaxPool2D(pool_size=(2, 2))(image_model_conv3)
#     image_model_batch_norm3 = BatchNormalization()(image_model_max_pool3)

#     # dropoput
#     images_model_dropout = Dropout(.3)(image_model_batch_norm3)

#     # mlp
#     images_model_dense = Dense(1024, activation='relu')(images_model_dropout)
#     image_model_batch_norm4 = BatchNormalization()(images_model_dense)

#     # output 
#     image_model_output = Dense(64, activation='relu')(image_model_batch_norm4)

#     # next create the model for the landmarks
#     landmark_model_input = Input(shape=(30, 1))

#     # first dense layer
#     landmarks_model_dense1 = Dense(1024, activation='relu')(landmark_model_input)
#     landmarks_model_batch_norm1 = BatchNormalization()(landmarks_model_dense1)

#     # output
#     landmarks_model_output = Dense(64, activation='relu')(landmarks_model_batch_norm1)

#     # concatenate the two models
#     concat = Concatenate()([image_model_output, landmarks_model_output])

#     # final output layer of the two models
#     output = Dense(7, activation='softmax')(concat)

#     # create the final model
#     model = Model(inputs=[images_model_input, landmark_model_input], outputs=[output])

#     return model