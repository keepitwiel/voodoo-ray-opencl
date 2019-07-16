# Improvements

## two-step tracing
  1. trace from camera to surface
  2. primary light: from surface, sample all nearby light sources
  3. secondary light: from surface, scatter rays and then sample all nearby light sources from secondary surface
      - can be done directly or via presampling
      
      
## intersections
  - given a set of squares and a line progressing from an origin, how do we determine 
    - which squares intersect the line;
    - which of those is closest to the origin;
    - at which point on that square does the line intersect;
    - is the square facing the opposite direction of the line's direction
  - many squares will share the same plane, so not necessary to check for each square where intersection happens
    - idea: identify all possible planes