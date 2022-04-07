docker build ./ -t uniformface --build-arg UNAME=$USER --build-arg UID=$(id -u $USER) --build-arg GID=$(id -g $USER)
