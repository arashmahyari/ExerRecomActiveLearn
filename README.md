# Real Time Learning from An Expert in Recommendation Systems with Marginal Distance Probability Distribution

**<em>Abstract:</em>** Recommendation systems play an important role in today's digital world. They have found applications in various applications such as music platforms, e.g., Spotify, and movie streaming services, e.g., Netflix. Less research effort has been devoted to physical exercise recommendation systems. On the other hand, sedentary life styles have become the major driver of several diseases as well as healthcare costs. In this paper, we develop a recommendation system for recommending daily exercise activities to users based on their history, profile and similar users. The developed recommendation system uses a deep recurrent neural network with user-profile attention and temporal attention mechanisms. 

Moreover, exercise recommendation systems are significantly different from streaming recommendation systems in that we are not able to collect click feedback from the participants in exercise recommendation systems. Thus, we propose a real-time, expert-in-the-loop active learning procedure. The active learners calculate the uncertainty of the recommender at each time step for each user and ask an expert for recommendation when the certainty is low. In this paper, we derive the probability distribution function of <em>marginal distance</em>, and use it to determine when to ask experts for feedback. Our experimental results on an mHealth dataset shows the improved accuracy after incorporating the real-time active learner with the recommendation system.


