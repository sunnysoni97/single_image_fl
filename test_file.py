from fed_df_data_loader.get_crops_dataloader import get_distill_imgloader

if __name__ == "__main__":
    path = "./data/single_img_crops/crops"
    dl = get_distill_imgloader(path)

    i=0
    for img,labels in dl:
        if(i<5):
            print(img[0].shape)
            print(labels[0])
        i+=1

    print(f'Total batches : {i}')
    
    
    

