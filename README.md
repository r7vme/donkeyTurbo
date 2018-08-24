# Donkey Turbo ( silly name, but who cares ;) )

Extension for donkeycar module for [deep-berlin.ai](https://deep-berlin.ai) challenge 2018.

Main differences from default donkey:
- Use cropped image 78x160. So image contains only the road.
- Linear speed computation depending on the angle.

# Quick start (on local machine)

Prepare python 3 virtual environment and install dependencies.
```
virtualenv env -p python3
source env/bin/activate
pip install -e .
```

Run training with
```
./manage.py train --model <output model path> --tub=<tubs path comma separated>
```

Run simulation server with
```
./manage.py sim --model=<path to model>
```

# Quick start (on Raspberry pi)

```
virtualenv env -p python3
source env/bin/activate
pip install -e . -i https://www.piwheels.org/simple

# Install Raspberry Pi specific dependecies
pip install "donkeycar[pi]==2.5.1" -i https://www.piwheels.org/simple
```

Run donkey car
```
./manage.py drive --model modeltmp
```
