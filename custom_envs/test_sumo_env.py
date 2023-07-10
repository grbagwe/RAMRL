import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


sumoBinary = "/home/gauravb/sumo/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "./sumo_ConfigParallelRamp/ramp_parallel.sumocfg"]

import traci
traci.start(sumoCmd)
step = 0
while step < 1000:
   traci.simulationStep()
   step += 1

traci.close()

