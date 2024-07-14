import numpy as np
import matplotlib.pyplot as plt

def search(x, y):
    factor = 0.1
    center = (5, 5)
    return np.exp(factor * (-np.square(x - center[0]) - np.square(y - center[1])))

def value(x, y, z):
    factor = 0.1
    center = (0, 5, 5)
    return np.exp(factor * (-np.square(x - center[0]) - np.square(y - center[1]) - np.square(z - center[2])))

def threat(x, y, z):
    factor = 1
    center = (0, 5, 0)
    return np.exp(factor * (-np.square(x - center[0]) - np.square(y - center[1]) - np.square(z - center[2])))

from scipy.optimize import minimize

grid = np.meshgrid(np.arange(10), np.arange(10))
search_grid = search(grid[0], grid[1])

def objective(waypoints):
    tot = 0
    for i in range(0, len(waypoints), 3):
        tot += -search_grid[int(min(np.round(waypoints[0]), 9)), int(min(np.round(waypoints[1]), 9))]
    return tot

# def eq_const(waypoints, i):
#     def eq(waypoints):
#         x = waypoints[i]
#         y = waypoints[i+1]
#         z = waypoints[i+2]
#         x2 = waypoints[i+3]
#         y2 = waypoints[i+4]
#         z2 = waypoints[i+5]
        
#         return np.sqrt(np.square(x2-x) + np.square(y2-y) + np.square(z2-z))-1
def get_eq_const(max_dist):

    def eq_const(waypoints, d=max_dist):
        tot = 0
        
        for v in range(0, len(waypoints)-5, 3):
            i=v
            x = waypoints[i]
            y = waypoints[i+1]
            z = waypoints[i+2]
            x2 = waypoints[i+3]
            y2 = waypoints[i+4]
            z2 = waypoints[i+5]

            dist = np.sqrt(np.square(x2-x) + np.square(y2-y) + np.square(z2-z))
            tot += dist
        
        return d - tot

    return {'type': 'ineq', 'fun': eq_const}




def equality_constraints(waypoints):
    all_constraints = []
    tot_dist = 0
    for v in range(0, len(waypoints)-5, 3):
        
        def eq_const(waypoints, i=v):
            x = waypoints[i]
            y = waypoints[i+1]
            z = waypoints[i+2]
            x2 = waypoints[i+3]
            y2 = waypoints[i+4]
            z2 = waypoints[i+5]

            # print(i)
            
            return np.sqrt(np.square(x2-x) + np.square(y2-y) + np.square(z2-z))-2

        all_constraints.append({'type': 'eq', 'fun': eq_const})

    return all_constraints

def init_constraints(waypoints, pos):
    consts = []
    for i in range(3):
        def init_const(waypoints, v=i, p = pos + [0]):
            # print(v)
            return waypoints[v] - p[v]
        consts.append({'type': 'eq', 'fun': init_const})
    return consts
    # return {'type': 'eq', 'fun': }

def final_constraints(waypoints):
    consts = []
    for i in range(3):
        def init_const(waypoints, v=i):
            # print(v)
            return waypoints[len(waypoints) - 3 + v]
        consts.append({'type': 'eq', 'fun': init_const})
    return consts


t = 0
xs = []
ys = []
zs = []
horizon = 10
exec = 5
max_dist = 15
pos = [0, 0]

while max_dist > 0:
    print(max_dist)
    init_vals = [0] * 3 * horizon

    dist_const = [get_eq_const(max_dist)]
    cs = init_constraints(pos)
    fs = final_constraints()
    consts = dist_const + cs + fs

    result = minimize(objective, init_vals, constraints=consts)
    solution = result['x']
    print(solution)

    curr_xs = []
    curr_ys = []
    for i in range(0, exec*3, 3):
        curr_xs.append(solution[i])
        curr_ys.append(solution[i+1])
        search_grid[int(min(np.round(curr_xs[i]), 9)), int(min(np.round(curr_ys[i]), 9))] = 0
    print(curr_xs, curr_ys)
    d = 0
    for i in range(len(curr_xs) - 1):
        d += np.sqrt(np.square(curr_xs[i+1] - curr_xs[i]) + np.square(curr_ys[i+1] - curr_ys[i]))
    print(d)
    max_dist -= d

    pos = [curr_xs[-1], curr_ys[-1]]
    
plt.imshow(search_grid)

plt.plot(xs, ys)
plt.colorbar()

