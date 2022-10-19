import numpy as np
import matplotlib.pyplot as plt

state = {
    'alertness': 0,
    'hypertension': 0,
    'intoxication': 0,
    'time_since_slept': 0,
    'time_elapsed': 0,
    'work_done': 0
}


def wakeup(state):
    state['alertness'] = np.random.uniform(0.7, 1.3)
    state['time_since_slept'] = 0

wakeup(state)
print(state)


def drink_coffee(state):
    state['alertness'] += np.random.uniform(0, 1)
    state['hypertension'] += np.random.uniform(0, 0.3)


drink_coffee(state)
print(state)


def drink_beer(state):
    state['alertness'] -= np.random.uniform(0, 0.5)
    state['hypertension'] += np.random.uniform(0, 0.3)
    state['intoxication'] += np.random.uniform(0.01, 0.03)


drink_beer(state)
print(state)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def heart_attack_risk(hypertension, heart_attack_proclivity=0.5):
    return heart_attack_proclivity * sigmoid(hypertension - 6)


plt.figure(figsize=(12,8))
xspace = np.linspace(0,10,100)
plt.xlabel('hypertension')
plt.ylabel('heart attack risk')
plt.plot(xspace, heart_attack_risk(xspace))
plt.show()


def heart_attack_occured(state, heart_attack_proclivity=0.5):
    return np.random.uniform(0, 1) < heart_attack_risk(state['hypertension'], heart_attack_proclivity)


print(heart_attack_occured(state))


def alertness_decay(time_since_slept):
    return sigmoid((time_since_slept - 40)/10)


plt.figure(figsize=(12,8))
xspace = np.linspace(0, 100, 100)
plt.xlabel('halfhours since slept')
plt.ylabel('alertness decay')
plt.plot(xspace, alertness_decay(xspace))
plt.show()

decay_rate = 0.97
half_life = decay_rate ** 24
print(half_life)


def half_hour_passed(state):
    state['alertness'] -= alertness_decay(state['time_since_slept'])
    state['hypertension'] = decay_rate * state['hypertension']
    state['intoxication'] = decay_rate * state['intoxication']
    state['time_since_slept'] += 1
    state['time_elapsed'] += 1


half_hour_passed(state)
print(state)


def crippling_anxiety(alertness):
    return sigmoid(alertness - 3)


plt.figure(figsize=(12,8))
xspace = np.linspace(0, 10, 100)
plt.xlabel('alertness')
plt.ylabel('crippling anxiety')
plt.plot(xspace, crippling_anxiety(xspace))
plt.show()


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def ballmer_function(intoxication):
    return sigmoid((0.05 - intoxication) * 50) + 2 * gaussian(intoxication, 0.135, 0.005)


plt.figure(figsize=(12,8))
xspace = np.linspace(0, 0.26, 1000)
plt.xlabel('blood alcohol content')
plt.ylabel('programming skill')
plt.plot(xspace, ballmer_function(xspace))
plt.show()


def productivity(state):
    p = 1
    p *= state['alertness']
    p *= 1 - crippling_anxiety(state['alertness'])
    p *= ballmer_function(state['intoxication'])
    return p


print(state)
productivity(state)


def work(state):
    state['work_done'] += productivity(state)
    half_hour_passed(state)


work(state)
print(state)
