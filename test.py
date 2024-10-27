from ukss_pi import UKSS_PI

def test(txt):
    kss = UKSS_PI(text=txt)
    kss.get_keywords()
    
    
# txt = """
#     Replace with a long sample text to test the UKSS-PI algorithm.
# """


# A sample text

txt = '''
The gym is a dedicated space for individuals to enhance their physical fitness, mental well-being, and personal health. Gyms offer a variety of equipment and facilities designed to cater to a wide range of fitness goals, from weightlifting and strength training to cardiovascular and flexibility exercises. Common equipment includes free weights, resistance machines, treadmills, rowing machines, and stationary bikes. Many gyms also feature open areas for stretching, functional training, and group exercise classes like yoga, Pilates, Zumba, and spin, making it accessible for people of all fitness levels and preferences. Working out at the gym provides numerous benefits beyond the physical. Regular exercise helps reduce stress by releasing endorphins, the body's natural mood enhancers, which improve emotional health and mental clarity. A routine gym visit fosters a sense of discipline and consistency, helping individuals build better habits over time. Additionally, the gym environment can be highly motivating, as it brings together people with similar goals, creating a sense of community and encouragement. Trainers and fitness professionals at the gym are often available to provide guidance, correct form, and develop personalized workout plans, making the gym experience safer and more effective. For many, the gym becomes a haven, a place where they can focus on themselves, escape daily pressures, and achieve gradual, measurable progress in their fitness journeys.
'''

test(txt=txt)