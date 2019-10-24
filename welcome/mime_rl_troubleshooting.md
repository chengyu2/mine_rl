Q: How come this line fails:
env = gym.make('MineRLNavigateDense-v0')
There is an error message:
Minecraft process finished unexpectedly. There was an error with Malmo.

A: 
The issue here is that you might have a different version of Java installed.

For Linux users:
```bash
echo $JAVA_HOME
# expect as output /usr/lib/jvm/java-1.8.0-openjdk-amd64
ls /usr/lib/jvm
# it will list all the java versions installed
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
```

For Mac User
```bash
# Reference : https://stackoverflow.com/questions/21964709/how-to-set-or-change-the-default-java-jdk-version-on-os-x
/usr/libexec/java_home -V
export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)`
```



