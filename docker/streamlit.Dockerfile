FROM python:3.10.1-slim

ARG USER_NAME=appuser
ARG USER_ID=1001

RUN groupadd -g $USER_ID $USER_NAME && \
    useradd -g $USER_ID -m -u $USER_ID -s /bin/bash $USER_NAME && \
    chgrp -R 0 /home/$USER_NAME && \
    chmod -R g=u /home/$USER_NAME

USER $USER_ID

WORKDIR /app

COPY --chown=$USER_ID:$USER_ID dashboard_requirements.txt .
RUN pip install -r dashboard_requirements.txt

COPY --chown=$USER_ID:$USER_ID . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
