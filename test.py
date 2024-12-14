from ukss_pi import UKSS_PI


def test(txt):
    kss = UKSS_PI(text=txt)
    result = kss.get_keywords()
    return result


# txt = """
#     Replace with a long sample text to test the UKSS-PI algorithm.
# """


# A sample text

txt = """
The gym is a dedicated space for individuals to enhance their physical fitness, mental well-being, and personal health. Gyms offer a variety of equipment and facilities designed to cater to a wide range of fitness goals, from weightlifting and strength training to cardiovascular and flexibility exercises. Common equipment includes free weights, resistance machines, treadmills, rowing machines, and stationary bikes. Many gyms also feature open areas for stretching, functional training, and group exercise classes like yoga, Pilates, Zumba, and spin, making it accessible for people of all fitness levels and preferences. Working out at the gym provides numerous benefits beyond the physical. Regular exercise helps reduce stress by releasing endorphins, the body's natural mood enhancers, which improve emotional health and mental clarity. A routine gym visit fosters a sense of discipline and consistency, helping individuals build better habits over time. Additionally, the gym environment can be highly motivating, as it brings together people with similar goals, creating a sense of community and encouragement. Trainers and fitness professionals at the gym are often available to provide guidance, correct form, and develop personalized workout plans, making the gym experience safer and more effective. For many, the gym becomes a haven, a place where they can focus on themselves, escape daily pressures, and achieve gradual, measurable progress in their fitness journeys.
"""

# txt = """
# Creating a web app using Flask is an exciting journey for anyone interested in web development. Flask is a lightweight Python framework that is both beginner-friendly and highly versatile, making it a popular choice among developers. Whether you're building a small personal project or a robust application, Flask provides the tools you need. The first step in creating a Flask web app is to install Flask on your system. Before diving into the development process, ensure that Python is installed, as Flask is built on Python. Once Flask is installed, you can start setting up your project. This involves creating a new project directory and organizing it to include folders for templates, static files like CSS and JavaScript, and your main application file. Flask uses the concept of routes to define how users interact with your application. Routes map specific URLs to functions in your application. For instance, you can create a homepage by defining a route that displays a simple welcome message. Templates come into play here, allowing you to create HTML files that Flask renders dynamically, ensuring your app has a clean and professional look. Another key aspect of Flask development is handling user inputs. Forms can be created to capture data from users, which Flask processes to perform tasks like saving data to a database or updating information. Flask seamlessly integrates with databases like SQLite, making it easy to manage and store data. Testing is a crucial part of the development process. Flask’s built-in server lets you run your application locally and see how it works in a browser. This allows you to experiment and refine your app before deploying it. When your app is complete and ready to share, Flask applications can be deployed to hosting platforms like Heroku, AWS, or Google Cloud. Deployment ensures your web app is accessible to users around the globe. In summary, creating a web app using Flask involves setting up your project, defining routes, integrating templates, handling user inputs, and deploying the application. With its simplicity and flexibility, Flask empowers you to bring your ideas to life and share them with the world.
# """

result = test(txt=txt)
tags = result["keywords"]
summary = result["summary"]

print("Keywords: ", tags)
print("Summary: ", summary)
