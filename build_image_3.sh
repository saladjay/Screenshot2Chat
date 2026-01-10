ssh-keyscan ***REMOVED*** > known_hosts



DOCKER_BUILDKIT=1 docker build \
       	--ssh default \
	--secret id=known_hosts,src=./known_hosts \
       	-t chat_analysis:v1.5 \
	. --load

rm -f known_hosts
