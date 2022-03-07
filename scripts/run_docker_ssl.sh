docker container run --gpus all --shm-size=8g -itd --rm name docker-ssl \
    --net="host" -v $PWD:/project -v /Users/moritaeiji/project/supervised_learning/data/ -p 8888:8888 docker-ssl bash