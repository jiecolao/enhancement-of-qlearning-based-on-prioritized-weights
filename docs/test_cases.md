# INTRAMUROS ENVIRONMENT

TEST SEPERATELY

## DEFAULT ENVIRONMENT PARAMETERS
```python
PRESET_ENVIRONMENT = PRESET_ENVIRONMENTS[1] # Intramuros Environment (20x20)

episodes = 2000
ep_tracker = 10
```

## TEST CASES
- [ ] Intramuros Environment
- [ ] Intramuros Environment with 3 dynamic obstacles
- [ ] Intramuros Environment with 5 dynamic obstacles
- [ ] Intramuros Environment with 9 dynamic obstacles

# FILENAME FORMAT
Intramuros_{no_of_obstacles}.txt
Intramuros_{no_of_obstacles}.jpg
Example:
Intramuros_3.txt
Intramuros_3.jpg

___
# DYNAMIC ENVIRONMENT

## DEFAULT ENVIRONMENT PARAMETERS
```python
PRESET_ENVIRONMENT = 
  PRESET_ENVIRONMENTS[3],
  PRESET_ENVIRONMENTS[4],
  PRESET_ENVIRONMENTS[5],
  PRESET_ENVIRONMENTS[6],
  PRESET_ENVIRONMENTS[7],
  PRESET_ENVIRONMENTS[8]
  
episodes = 1000
ep_tracker = 10
no_of_obstacles = 3 -> 5 > 9
is_dynamic_obs = True
```

## TEST CASES

- [] 10x10 with 3 Dynamic Obstacles
- [] 10x10 with 5 Dynamic Obstacles
- [] 10x10 with 9 Dynamic Obstacles
- [] 20x20 with 12 Dynamic Obstacles
- [] 20x20 with 29 Dynamic Obstacles
- [] 20x20 with 36 Dynamic Obstacles
- [] 30x30 with 27 Dynamic Obstacles
- [] 30x30 with 45 Dynamic Obstacles
- [] 30x30 with 81 Dynamic Obstacles
- [] 40x40 with 48 Dynamic Obstacles
- [] 40x40 with 80 Dynamic Obstacles
- [] 40x40 with 144 Dynamic Obstacles
- [] 50x50 with 75 Dynamic Obstacles
- [] 50x50 with 125 Dynamic Obstacles
- [] 50x50 with 225 Dynamic Obstacles
- [] 75x75 with 169 Dynamic Obstacles
- [] 75x75 with 281 Dynamic Obstacles
- [] 75x75 with 506 Dynamic Obstacles
- [] 100x100 with 300 Dynamic Obstacles
- [] 100x100 with 500 Dynamic Obstacles
- [] 100x100 with 900 Dynamic Obstacles

## FILENAME FORMAT
Dynamic_{grid}_{no_of_obstacles}.txt
Dynamic_{grid}_{no_of_obstacles}.jpg

Example: 
Dynamic_10x10_3.txt
Dynamic_10x10_3.jpg

___
# OBSTACLE-FREE ENVIRONMENT

## DEFAULT ENVIRONMENT PARAMETERS
```python
PRESET_ENVIRONMENT = 
  PRESET_ENVIRONMENTS[3],
  PRESET_ENVIRONMENTS[4],
  PRESET_ENVIRONMENTS[5],
  PRESET_ENVIRONMENTS[6],
  PRESET_ENVIRONMENTS[7],
  PRESET_ENVIRONMENTS[8]

episodes = 1000
ep_tracker = 10
no_of_obstacles = 0
is_dynamic_obs = False
```

## TEST CASES
- [] 10x10
- [] 20x20
- [] 30x30 
- [] 40x40
- [] 50x50
- [] 75x75
- [] 100x100

## FILENAME FORMAT
Free_{grid}_{no_of_obstacles}.txt
Free_{grid}_{no_of_obstacles}.jpg
Ex. 
Free_10x10_0.txt
Free_10x10_0.jpg