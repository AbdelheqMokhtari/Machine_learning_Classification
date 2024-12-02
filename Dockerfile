FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

# Run all scripts in order
CMD ["./run_scripts.sh"]