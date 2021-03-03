# Robot Voice Command

This module allows the use of custom voice commands on a robot (such as the Jetson Nano) as well as any automated system that has a microphone and actuation outputs. BERT is used in the model to provide semantic context to speech. For example, instead of simply recognizing the phrase 'go left', an entire sentence command can be given with contextual cues. This allows for alot more flexibilty than simply performing a convolution operation with a canned input command.

The setup works as follows:
A speech recognition module converts speech into text which is passed to a passive listener. When the passive listener hears a keyword for waking the robot (e.g. "Okay robot"), active listening starts. A confirmation is also given to the user which is currently in the form of speech via Google text to speech. This can also be coupled with an LED flag activated from a GPIO pin. 

After active listening starts, speech recognition starts again and inputs the provided sentence into the natural language processing model. The model interprets the command and provides an estimate of actuator output. This output can then be sent to the robot to move. Below is a visual outline:

![outline](https://user-images.githubusercontent.com/74885742/109883374-52e30a80-7c49-11eb-97f1-a2d7f3ebd046.jpg)

Improvements to make:
1. Speed up the speech recognition after active listening is enabled
2. Retreive model predictions faster - might need to remove BERT and make custom transformers.
3. Add more training data as well as new sets of commands to training data.
