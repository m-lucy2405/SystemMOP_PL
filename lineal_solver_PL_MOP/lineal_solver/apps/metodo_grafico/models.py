from django.db import models

class GraphicalModel(models.Model):
    # Define fields for the graphical model
    name = models.CharField(max_length=100)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class GraphicalSolution(models.Model):
    # Define fields for storing graphical solutions
    model = models.ForeignKey(GraphicalModel, on_delete=models.CASCADE)
    solution_data = models.JSONField()  # Store solution data in JSON format
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Solution for {self.model.name}'