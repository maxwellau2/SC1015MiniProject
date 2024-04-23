### Business Proposition
#### In order to quantiy even stating a data science project, we have to ask ourselves what problem we are trying to solve. There is little to no point in creating a complex model that **does not** solve a problem

1. Why is PM2.5 bad?
According to the American Heart Association:

*"Exposure to PM <2.5 μm in diameter (PM2.5) over a few hours to weeks can trigger cardiovascular disease-related mortality and nonfatal events; longer-term exposure (eg, a few years) increases the risk for cardiovascular mortality to an even greater extent than exposures over a few days and reduces life expectancy within more highly exposed segments of the population by several months to a few years."*
<a href="https://blissair.com/what-is-pm-2-5.htm">link here</a>

2. What are we trying to solve?
- Although we are unable to implement solutions to counter overall air pollution in the globe (come on, we are just university students, not UNICEF), we can ideate systems to mitigate exposure to PM2.5 for people in closed areas by creating a **Simple Reflex Agent with State(SRAS)**.
- This **SRAS** will switch on air conditioning devices, air purifying systems, and send reminder messages to relevant parties to wear masks.
- However, the SRAS needs to be driven by data, and hard coding values are simply not enough to **warn people about the future (next 1 hour)**. People simply won't have the level of readiness if informed at such a short notice.
- Futhermore, what happens when the PM2.5 sensor is down? Can we implement a **system of redundancy** such that when the PM2.5 sensor is down, we are able to estimate it within an acceptable margin?
- We will use a data driven approach to design this **SRAS** using content covered in the course, and beyond.