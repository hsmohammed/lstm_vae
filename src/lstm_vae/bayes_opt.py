
@use_named_args(dimensions=dimensions)
def fitness(intermediate_dim, latent_dim, learning_rate, momentum):
    """
    Hyper-parameters:
    intermediate_dim:     Intermediate dimensions.
    latent_dim:  Latent Dimensions.
    """

    # Print the hyper-parameters.
    print('intermediate diminsion: {0:.1e}'.format(intermediate_dim))
    print('latent_dim:', latent_dim)
    print()
    
    # Create the neural network with these hyper-parameters.
    K.clear_session()

    model = VAE_def(intermediate_dim = intermediate_dim, latent_dim = latent_dim, learning_rate = learning_rate, momentum = momentum)

    # Dir-name for the TensorBoard log-files.
    log_dir = './log1'
    
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=1,
        write_graph=True,
        write_grads=False,
        write_images=False)
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    
    filepath="save_model/LPT-{epoch:02d}-{loss:.4f}.h5"
    save_callback = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True,monitor='val_loss',save_best_only=False, mode='auto', period=1)

    
    # def scheduler(epoch,lr):
        
    #     return lr*(0.99**(epoch//10+1))
                   
        
    # lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler,verbose=1)
    
    class customcallback(keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            print(logs.keys())
   
    # Use Keras to train the model.
    

    
    def train_data_generator(X = X_train1):
        while True:
            X1 = X
            for i in range(len(X1)):
                a = X1[0]
                
                b = (np.array([a]),np.array([a]))
                yield b
                X1 = X1[1:]
                
    def validate_data_generator(X = X_validate):
        while True:
            X1 = X
            for i in range(len(X1)):
                a = X1[0]
                
                b = (np.array([a]),np.array([a]))
                yield b
                X1 = X1[1:]
                
                
    train_generator = train_data_generator()
            
    validation_generator = validate_data_generator()
    

    history = model.fit(train_generator,
                        epochs=20,
                        steps_per_epoch=len(X_train1),
                        validation_data=validation_generator,
                        validation_steps=len(X_validate),
                        callbacks=[customcallback(), callback_log, save_callback,PlotLossesKeras()])
    accuracy = history.history['val_loss'][-1]


    del model
    
    K.clear_session()
    
    return accuracy
