python -m SimpleHTTPServer 8181
echo http://localhost:8181/presentation/
setsid firefox http://localhost:8181/presentation/ &
