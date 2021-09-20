Installing Tensorflow_macos m1 version 
======================================

1. Download environment.yml from this site:

https://raw.githubusercontent.com/mwidjaja1/DSOnMacARM/main/environment.yml


2. Follow all checks and install 2.0rc1 version from here:

https://github.com/apple/tensorflow_macos/issues/153

3. STOP ...here if you want to stick to mac -gpu support...

If you go to the next step, you get Tensorflow newer version, but will loose mac m1 gpu support.



4. Then install 2.5 by doing:

python -m pip install tensorflow-macos --upgrade --force --no-dependencies

5.>python

> import tensorflow as tf.

You should not see any errors...



