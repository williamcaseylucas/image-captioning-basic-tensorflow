for i in range(epochs):
    generator = data_generator(
        train_ids,
        img_id_to_captions,
        features,
        tokenizer,
        max_length,
        vocab_size,
        batch_size,
    )
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
