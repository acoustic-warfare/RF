﻿# Python agent

## Introduction
This repo is based on the python examples provided by WARA-PS at https://github.com/wara-ps/waraps-agent-examples.git . For more information about how the agents work, checkout https://api.docs.waraps.org

This repo contains code to create virtual python agents which can be used on the 2022 Arena map and the Integration map. 

Note that the code influencing the python agents behaviour make it act as a drone would

##### Install Dependencies:
##Getting started
#### Install:
To run the code for the python agents you need certain libraries which you can install through the command:
```pipenv install -r requirements.txt```

##### Prerequisites:  
If you don't use Python 3.10 or newer, you need to install 'pip' separately using the command ```python get-pip.py``` 
as well as 'pipenv' using ```pip3 install pipenv```

##### Run:  
```pipenv run python main.py```
####Code edits:
To stream stream the agent to the 2022 Arena map and the Integration map and not a local host you will need to a username and a password.

#### Docker
The start_pluto.sh file in the RF-directory can be used to run the repo through docker instead. 
