import gym

mountaincar = gym.make('MountainCarContinuous-v0')
observation = mountaincar.reset()
location, speed = observation

_ = mountaincar.observation_space.sample()

print(_)

location_low, speed_low = mountaincar.observation_space.low
location_high, speed_high = mountaincar.observation_space.high

print(f'Car location can range from {location_low} to {location_high}')
print(f'Car speed can range from {speed_low} to {speed_high}')