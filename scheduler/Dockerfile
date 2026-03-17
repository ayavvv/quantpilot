FROM docker:27-cli
RUN apk add --no-cache bash openssh-client tzdata
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
COPY scheduler/scripts/ /scripts/
COPY scheduler/crontab /etc/crontabs/root
RUN chmod +x /scripts/*.sh
CMD ["crond", "-f", "-l", "2"]
