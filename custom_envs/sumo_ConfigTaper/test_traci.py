import os,sys


if 'SUMO_HOME' in os.environ:
    SUMO_HOME = os.environ['SUMO_HOME']
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    print('sumo_loaded')
tools = os.path.join(SUMO_HOME, 'tools')
sys.path.append(tools)
print(tools)

import traci as traci
import sumolib
LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ

sumoBinary = os.path.join(SUMO_HOME, "bin/sumo-gui")


sumoConfigFile = "./custom_envs/sumo_ConfigParallelRamp/ramp_parallel.sumocfg"
sumo_cmd = [sumoBinary,
                     '-n', './custom_envs/sumo_ConfigParallelRamp/ramp_parallel.net.xml',
                     '-r', './sumo_ConfigParallelRamp/ramp_parallel.rou.xml',
                     '--waiting-time-memory', '10000',
                     '--time-to-teleport', '-1', '--random']
traci.start(sumo_cmd)

