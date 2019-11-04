proposed algorithm: pseudocode
---

```
given:
    position, 
    direction, 
    environment[], 
    global_light[],
    intensity=INITIAL_INTENSITY, 
    flag=PROPAGATE:

while (flag != END):
    if flag == PROPAGATE:
        flag, color, position, next_block, intensity = propagate(
            position, 
            direction, 
            intensity, 
            INITIAL_LENGHT,
            environment[])

    elif flag == GLOBAL_LIGHT:
        intensity = get_intensity_from_global_light(direction, global_light[])
        flag = END

    elif flag == LOCAL_LIGHT:
        flag = END
         
    elif flag == WALL:
        intensity *= color
        direction = diffuse(position, next_block)
        flag = PROPAGATE
        
    elif flag == EARLY_TERMINATION:
        flag = END
    
return intensity;
```