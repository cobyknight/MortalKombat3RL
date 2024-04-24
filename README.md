How to get the AI to run:

This is a real pain in the ass to get to run. It uses a game emulator called gym retro that was developed by open ai and has not been updated since 2020 so there is a lot of downgrading that needs to be done to get this to run. We used VSCode to build this so this tutorial will have that in mind

First step is to uninstall whatever version of python you have installed and install python 3.7.9. I found the quickest one to get working is off the microsoft store

Type python into the console to verify you have the correct one installed.

Use the command “pip install retro”

Next you have to use the command “pip install gym==0.17.3” this should downgrade gym that was installed when you used the command retro

“Pip install stable_baselines3”

Next, put the mortal kombat emulator in the ai program file.

You then use “python -m retro.import” to import the game into the environment


