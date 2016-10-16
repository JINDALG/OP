# Procfile with nginx, pgbouncer, uWSGI and django-q
web: uwsgi uwsgi.ini
worker: celery -A face worker -l info
