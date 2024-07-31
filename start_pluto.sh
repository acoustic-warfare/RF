#!/bin/bash

# Reset
Color_Off='\033[0m'       # Text Reset

# Regular Colors
Black='\033[0;30m'        # Black
Red='\033[0;31m'          # Red
Green='\033[0;32m'        # Green
Yellow='\033[0;33m'       # Yellow
Blue='\033[0;34m'         # Blue
Purple='\033[0;35m'       # Purple
Cyan='\033[0;36m'         # Cyan
White='\033[0;37m'        # White

# Checking that user do not run as root
if [ "$EUID" -eq 0 ]
  then echo "Please do not run as root"
  exit
fi

# Checking docker installation
if ! [ -x "$(command -v docker)" ]; then
    echo 'Error: docker is not installed.' >&2
    read -p "Do you want to install it? (y/N)" choice
    case "$choice" in 
    y|Y ) install_docker;;
    * ) echo "Exiting" && exit 1;;
    esac
fi

# Function to check if a Docker Compose project is running
is_compose_running() {
  docker-compose ps -q | grep -q '[[:alnum:]]'
}

# Function to stop and remove Docker Compose services
remove_compose_services() {
  echo "Stopping and removing existing Docker Compose services..."
  docker-compose down -v --remove-orphans
}

# Check if Docker Compose is already running
if is_compose_running; then
  remove_compose_services
fi

# Grant local Docker containers permission to connect to the X server
xhost +local:docker

# Starting the container
docker-compose up --build pluto-app 