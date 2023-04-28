from fed_df_data_loader.split_standard import create_std_distill_loader

if __name__ == "__main__":
    test_dataloader = create_std_distill_loader(dataset_name='cifar100',storage_path='./data',n_images=1000)

    imgs, labels = next(iter(test_dataloader))
    print(len(imgs))
    print(len(labels))
    print(imgs[0].shape)
    print(labels[0])
    print("It worked!")