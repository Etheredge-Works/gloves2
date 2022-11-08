# docker buildx build --platform linux/amd64 -t etheredgeb/gloves2 --push .
docker buildx build --platform linux/amd64,linux/arm64 -t etheredgeb/gloves2 --push .