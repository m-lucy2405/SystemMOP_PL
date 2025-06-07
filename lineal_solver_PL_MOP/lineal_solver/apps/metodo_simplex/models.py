from django.db import models

class LinearProgrammingProblem(models.Model):
    objective_function = models.TextField()
    constraints = models.JSONField()
    solution = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"LP Problem: {self.objective_function}"


class Solution(models.Model):
    problem = models.ForeignKey(LinearProgrammingProblem, related_name='solutions', on_delete=models.CASCADE)
    variable_values = models.JSONField()
    optimal_value = models.FloatField()

    def __str__(self):
        return f"Solution for {self.problem}"