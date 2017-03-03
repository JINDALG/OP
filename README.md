# OP
Install pip:
	apt-get update
	apt-get install python-pip

Install virtualenv:
	pip install --upgrade virtualenv
	mkdir ~/.virtualenvs

Install virtualenvwrapper:
	pip install virtualenvwrapper

Set WORKON_HOME to your virtualenv dir:
	export WORKON_HOME=~/.virtualenvs

Edit .bashrc:
	Add this line to the end of ~/.bashrc so that the virtualenvwrapper commands are loaded.

	. /usr/local/bin/virtualenvwrapper.sh
	(How:
	Type "nano ~/.bashrc" in terminal and press enter.(without quotes)
	Press downward arrow to reach bottom of the file.
	Type ". /usr/local/bin/virtualenvwrapper.sh"
	Press Ctrl + x
	Press y
	press Enter
	)
	Exit and re-open your shell, or reload .bashrc with the command ". .bashrc"

Create your virtualenv:
	mkvirtualenv facerec
	(For working on your virtualenv, use workon facerec)

Install numpy, scipy, pandas, scikit-learn, nose, nolearn, django, etc:
	pip install numpy scipy pandas scikit-learn nose nolearn django celery django-admin-tools django-braces django-celery jsonfield kombu redis simplejson

Install boost python:
	apt-get install libboost-all-dev

Install cmake:
	add-apt-repository ppa:george-edison55/cmake-3.x
	apt-get update
	apt-get install cmake

Install dlib:
	pip install dlib

Install opencv:
	wget https://raw.githubusercontent.com/milq/milq/master/scripts/bash/install-opencv.sh
	bash install-opencv.sh
	export PYTHONPATH=/usr/local/lib/python2.7/site-packages/

Install torch:
	git clone https://github.com/torch/distro.git ~/torch --recursive
	cd ~/torch; bash install-deps;
	./install.sh
	source ~/.bashrc	
		
Install few more lua stuff:
	for NAME in dpnn nn optim optnet csvigo cutorch cunn fblualib torchx tds; do luarocks install $NAME; done

