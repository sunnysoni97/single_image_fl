CLIENT_GPUS=0.5
CLIENT_CPUS=2
SERVER_CPUS=12
TOTAL_CPUS=12
TOTAL_GPUS=1
TOTAL_MEM=13

STRATEGY=feddf
MODEL_NAME=resnet8

NUM_CLIENTS=2
NUM_ROUNDS=2
FRACTION_FIT=1.0
FRACTION_EVALUATE=0.0

DATASET_NAME=organamnist
PARTITION_ALPHA=1.0
PARTITION_VAL_RATIO=0.1

BATCH_SIZE=1024
LOCAL_EPOCHS=10
LOCAL_LR=0.05
DISTILL_BATCH_SIZE=1024
SERVER_LR=0.005
SERVER_STEPS=50
SERVER_EARLY_STEPS=1000
USE_EARLY_STOPPING=False
USE_ADAPTIVE_LR=False

SEED=42
CUDA_DETERMINISTIC=False

USE_CROPS=True
IMG_NAME=ct_scan_big.png
DISTILL_DATASET=cifar100
DISTILL_ALPHA=1.0
NUM_DISTILL_IMAGES=5000
DISTILL_TRANSFORMS=v0

WARM_START=True
WARM_START_ROUNDS=1
WARM_START_INTERVAL=5

KMEANS_N_CLUSTERS=10
KMEANS_HEURISTICS=easy
KMEANS_MIXED_FACTOR="50-50"

CLIPPING_FACTOR=3

DATA_DIR=./data
OUT_DIR='./results/out'
DEBUG=False

while getopts "l::g::c::" flag
do
    case "${flag}" in
        l) LOCAL_LR=${OPTARG};;
        g) SERVER_LR=${OPTARG};;
        c) CLIPPING_FACTOR=${OPTARG};;
    esac
done

echo "-----BEGIN-----"
echo "-----SETTINGS-----"
echo "STRATEGY:$STRATEGY"
echo "MODEL_NAME:$MODEL_NAME"

echo "NUM_CLIENTS:$NUM_CLIENTS"
echo "NUM_ROUNDS:$NUM_ROUNDS"
echo "FRACTION_FIT:$FRACTION_FIT"
echo "FRACTION_EVALUATE:$FRACTION_EVALUATE"

echo "DATASET_NAME:$DATASET_NAME"
echo "PARTITION_ALPHA:$PARTITION_ALPHA"
echo "PARTITION_VAL_RATIO:$PARTITION_VAL_RATIO"

echo "BATCH_SIZE:$BATCH_SIZE"
echo "LOCAL_EPOCHS:$LOCAL_EPOCHS"
echo "LOCAL_LR:$LOCAL_LR"
echo "DISTILL_BATCH_SIZE:$DISTILL_BATCH_SIZE"
echo "SERVER_LR:$SERVER_LR"
echo "SERVER_STEPS:$SERVER_STEPS"
echo "SERVER_EARLY_STEPS:$SERVER_EARLY_STEPS"
echo "USE_EARLY_STOPPING:$USE_EARLY_STOPPING"
echo "USE_ADAPTIVE_LR:$USE_ADAPTIVE_LR"

echo "SEED:$SEED"
echo "CUDA_DETERMINISTIC:$CUDA_DETERMINISTIC"

echo "USE_CROPS:$USE_CROPS"
echo "IMG_NAME:$IMG_NAME"
echo "DISTILL_DATASET:$DISTILL_DATASET"
echo "DISTILL_ALPHA:$DISTILL_ALPHA"
echo "NUM_DISTILL_IMAGES:$NUM_DISTILL_IMAGES"
echo "DISTILL_TRANSFORMS:$DISTILL_TRANSFORMS"

echo "WARM_START:$WARM_START"
echo "WARM_START_ROUNDS:$WARM_START_ROUNDS"
echo "WARM_START_INTERVAL:$WARM_START_INTERVAL"

echo "KMEANS_N_CLUSTER:$KMEANS_N_CLUSTERS"
echo "KMEANS_HEURISTICS:$KMEANS_HEURISTICS"
echo "KMEANS_MIXED_FACTOR:$KMEANS_MIXED_FACTOR"

echo "CLIPPING_FACTOR:$CLIPPING_FACTOR"

echo "-----SETTINGS END-----"

echo "-----EXPERIMENT BEGINS-----"

if [ $USE_CROPS == "True" -a $STRATEGY == "feddf" ] 
then
    echo "---------"
    echo "Generating crops for FedDF"
    python ./make_single_img_dataset.py --targetpath $DATA_DIR --num_imgs 20000 --seed $SEED --imgpath "./static/single_images/$IMG_NAME" --threads 18
    echo "---------"
fi
    
echo "Simulating $STRATEGY training"

python ./simulate.py --fed_strategy $STRATEGY --model_name $MODEL_NAME\
    --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --fraction_fit $FRACTION_FIT --fraction_evaluate $FRACTION_EVALUATE\
    --dataset_name $DATASET_NAME --data_dir $DATA_DIR --partition_alpha $PARTITION_ALPHA --partition_val_ratio $PARTITION_VAL_RATIO\
    --client_gpus $CLIENT_GPUS --client_cpus $CLIENT_CPUS --server_cpus $SERVER_CPUS --total_cpus $TOTAL_CPUS --total_gpus $TOTAL_GPUS --total_mem $TOTAL_MEM\
    --batch_size $BATCH_SIZE --local_epochs $LOCAL_EPOCHS --local_lr $LOCAL_LR\
    --distill_batch_size $DISTILL_BATCH_SIZE --server_lr $SERVER_LR --server_steps $SERVER_STEPS --server_early_steps $SERVER_EARLY_STEPS --distill_transforms $DISTILL_TRANSFORMS\
    --use_early_stopping $USE_EARLY_STOPPING --use_adaptive_lr $USE_ADAPTIVE_LR\
    --seed $SEED --cuda_deterministic $CUDA_DETERMINISTIC\
    --use_crops $USE_CROPS --distill_dataset $DISTILL_DATASET --distill_alpha $DISTILL_ALPHA --num_distill_images $NUM_DISTILL_IMAGES\
    --warm_start $WARM_START --warm_start_rounds $WARM_START_ROUNDS --warm_start_interval $WARM_START_INTERVAL\
    --kmeans_n_clusters $KMEANS_N_CLUSTERS --kmeans_heuristics $KMEANS_HEURISTICS --kmeans_mixed_factor $KMEANS_MIXED_FACTOR\
    --clipping_factor $CLIPPING_FACTOR\
    --out_dir $OUT_DIR --debug $DEBUG

echo "-----EXPERIMENT ENDS-----"
echo "-----END-----"
