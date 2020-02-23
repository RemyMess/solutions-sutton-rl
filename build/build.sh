echo "AUTHOR: 'Installing now packages for the repo in a virtual environment.'"
sudo apt-get update
sudo apt-get install python3-pip
sudo pip3 install virtualenv 

if ! /usr/bin/git pull; then
    echo "Failed to git pull..."
    exit
fi

if ! [ -d ".env" ]; then
    virtualenv -p python3 ../..env
fi
source ../.env/bin/activate

pip install -r requirements.txt
