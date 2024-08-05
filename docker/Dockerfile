# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /temp_dir

# Copy the requirements file into the container at /app
COPY ../requirements.txt /temp_dir/requirements.txt

# Install any dependencies
RUN pip install --no-cache-dir -r /temp_dir/requirements.txt

# Copy the rest of the application code into the container
COPY .. /temp_dir

# Expose port 5000 to the outside world
EXPOSE 5000

# Run the Flask server
CMD ["python", "-m", "src.app.app"]
