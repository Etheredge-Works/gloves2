# docker buildx build --platform linux/amd64 -t etheredgeb/gloves2 --push .
docker buildx build --platform linux/amd64,linux/arm64,linux/arm/v7 -t etheredgeb/gloves2:$(date +%s) --push .