[TOC]



# Overview

## The * Layer Internet Model

| layer       | application                                                  | protocol             |
| ----------- | ------------------------------------------------------------ | -------------------- |
| Application | 两个应用间的双向可靠字节流传输，例如http/bit-torrent         | HTTP/SMTP/SSH/FTP    |
| Transport   | TCP/IP保证数据可靠传输；保证正确数据传输顺序正确；<br />UDP  | TCP/UDP/RTP          |
| Network     | IP is unreliable and makes no guarantees；没有可靠数据传输的保证 | IP                   |
| Link        | 通过host和router之间或router之间的单个link来传送数据         | Ethernet/wifi/DSL/3G |
| Physical    |                                                              |                      |

数据传输：

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302140130079.png" alt="image-20220928160445181" style="zoom:50%;" />

7层模型：

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302140130863.png" alt="image-20220928160652649" style="zoom: 33%;" />

## 

# Applications

## 1. Principles of network applications

### 1.1 Network application architectures

*Summary: architectures for **applications** , not network*

*Conclusion：*

*应用程序的主流体系架构有client-server与P2P两种。*

1. *在client-server中，存在有**client**、**server**、多台dedicated servers组成的**data center**几种host，客户端之间不能直接通信，服务端是固定的、等待连接的；采用data center集成服务器的好处是有利于响应客户端的所有要求，减少单个服务器的拥堵/过载（集成服务，一对一->多对多），坏处是提高了带宽代价（servers间的反复互连）。*eg. FTP/Web/E-MAIL
2. *在P2P中，客户端之间进行直连通信，其优点为可扩展性、（对服务器，因为几乎不依赖服务器）低带宽代价，其缺点为安全性低（直连而不通过服务器显然隐蔽性低）、低性能（高度去中心化结构，散点结构）。eg. Skype/迅雷/BitTorrent。*

优缺点的评价维度如下：可扩展、集成服务、带宽代价、安全性、性能

|                   | client-server | P2P  |
| ----------------- | ------------- | ---- |
| scalability       |               | high |
| 集成服务          | high          | low  |
| low bandwith cost | No            | Yes  |
| security          | high          | low  |
| performance       | high          | low  |

我的理解是这样，P2P和中心化网络、分布式网络的区别应该在P2P是host-host，中心化是存在一个data center/中心服务器，分布式是存在多个分布的data center。改天问下chatgpt。。

#### client-server architecture(client, server, data center, advantages and shortages, examples)

Hosts: client, server, data center(multiple servers)

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202211111508988.png" alt="image-20221111150843862" style="zoom:50%;" />

1. server:
   1. fixed address 固定地址
   2. always-on 等待连接
2. client: 
   1. not directly communicate with each other 客户端之间不能直接相互通信
3. data center:
   1. including multiple servers 多个服务器组成
   2. advantage: solve the problem of incapability of keeping up with all requests 响应需求↑
   3. shortage: must pay recurring interconnection and bandwidth costs(with the designs with data centers) 带宽代价↑
4. examples: Web, FTP, Telnet, E-MAIL

#### P2P architecture

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202211112311789.png" alt="image-20221111231152657" style="zoom:50%;" />

1. peer-to-peer: directly, without passing through a dedicated server 客户端直连
3. advantage
   1. self-scalability 可扩展
   2. cost effective, not require significant server infrastructure and server bandwidth 服务器带宽/代价↓
3. shortage: security, performance and reliability due to their highly decentralized structure 安全性、性能↓
5. example: BitTorrent, Xunlei, Skype

### 1.2 Processes Communicating

*Summary: processes running on different hosts(potentially on different operating systems) communicate **through application layer***

*Conclusion: application间的通信本质上是进程间通信，在不同OS、host上运行的进程通过application layer通信。一般将通信双方的进程**标记为client/server**，一个Web process可以既是client（browser process，download operation）也是server（server process，upload operation），该标记是人为的、非绝对的。要进行通信，首先要**确定双方进程的地址，通过 [IP: port number]的形式定位**，IP定位host，端口号定位该host上运行的process；在通信过程中，利用每个host上传输层与应用层/进程与网络之间的**接口Socket**。对于应用层开发者来说，其能控制的是**Socket中所有应用层的部分**，以及**传输层协议及其部分参数**（最大缓冲容量、最大段容量），该application将会建立在开发者选择的传输层协议所提供的服务之上。*

#### object

1. label: client(download) process /& server(upload) process
2. Identify Processes: use IP addresses(32 bits, identify the host) and port number(identify the process) to address processes(example: *[IP: port number],[localhost:8080]*)

examples:

1. a Web server is identified by port number 80. 

2. A mail server process (using the SMTP protocol) is identified by port number 25.

#### interface

interface between process(application) and network, between application layer and transport layer within a host: **socket (API)**

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202211112339983.png" alt="image-20221111233944866" style="zoom:50%;" />

#### what application developer controls

1. on the application layer side: everything  of the socket （socket中的应用层部分
2. on the transport layer side
   1. which **transport protocol** to use，传输层协议
   2. the capability to fix a few transport-layer **parameters**(maximum buffer, maximum segment sizes) 参数，最大缓冲，最大段
   3. the application is built on **transport-layer services** provided by that protocol 该应用将建立在该协议提供的传输层服务上





### 1.3 Transport Services Available to Applications

*Conclusion：传输层协议能为应用程序提供**可靠数据传输、吞吐量、实时性、安全性**四个维度的服务。应用层将消息存入当前进程的socket，传输层将该消息发送给目标接收进程的socket。*

应用层send messages through local socket，传输层send it to the socket of destination process

应用层request，传输层offer/ensure；~~什么甲乙方关系~~

#### 4 dimensions

传输协议能提供service of 4 dimensions:

1. reliable data transfer（not necessary for loss-tolerant applications, loss/data integrity/order)
2. throughput(specific requirements for bandwidth-sensitive applications，比如视频切换清晰度，应用层可以切换编码方式来降低对带宽的需求; contrary: elastic applications)
3. timing 实时性，与吞吐量有关
4. security

### 1.4 Transport Services Provided by the Internet

传输层协议及其提供的服务有TCP/UDP两种。**TCP是面向连接的、需要三次握手的、全双工的、可靠数据传输的、有拥堵控制的；但它与UDP都没有加密技术（安全性缺陷）**，因此出现了SSL（Secure socket layer），一种加入加密、端点鉴别、数据完整性技术的改良TCP协议，具体而言就是在原本的TCP通信中添加一层加密/解密socket层（这意味着它有独立的socket API），client先在SSL socket加密再传送给TCP socket，server先解密再传送给TCP socket。当今的网络能够支持时间敏感的应用，但无法支持任何对实时性与吞吐量的保证（guarantee）。

*这些提供的服务与上一小节中的四个维度是对应的。*

#### TCP

1. connection-oriented, handshaking, full-duplex connection
2. reliable data transfer
3. congestion-control(throughput, timing)
4. no encryption(security)

#### Secure Sockets Layer(SSL)

SSL是对TCP的改良(TCP-enhanced-with-SSL)，而非一种独立的网络传输协议，其服务包括encryption, data integrity, end-point authentication端点鉴别。

1. SSL服务代码需要被写在client和server两端的应用层
2. 有独立的socket API。
3. 过程：client-SSL socket(client encryption)-client TCP socket-server TCP socket-SSL socket(server decryption)-server TCP socket-server


#### UDP

1. connectionless, no handshaking
2. unreliable data transfer
3. no congestion-control mechanism

### 1.5 Application-Layer Protocols

*Conclusion: 应用层协议是网络应用的组成之一，决定了不同终端系统上的进程如何通信，具体决定了交换的信息类型、语法、字段语义、发送响应的规则。这些协议有些是定义在RFCs中的共有协议，如HTTP，有些则是私有的，如Skype使用的协议。*

#### what a protocol determines

1. 交换的信息类型，request & response
2. syntax，消息类型的语法，不同字段field的意思
3. when & how 发送消息& response

#### public/proprietary

1. public：一些应用层协议在RFC(请求注解，Request For Comments)中指定，因此属于公共领域。例如HTTP RFC
2. proprietary：其余的则是私有的，例如Skype

$application\ layer\ protocol \in application$，一个client-server应用程序可以包括：client，server，application layer protocol，standards of content format 

web-mail-DNS-directory service video streaming-P2P



## 2. The Web and HTTP

the World Wide Web -- first Internet application

### 2.1 HTTP(Hyper Text Transfer Protocol)

*Conclusion: HTTP(Hyper Text Transfer Protocol)超文本传输协议是Web的应用层协议，其为stateless protocol，不记录client的信息。Web是client-server架构的应用程序，其采用TCP传输层协议，client与server之间利用TCP socket进行通信，server将Web page中的objects(HTML files/images etc.)传送给client作为响应。*

HTTP, the application-layer protocol of Web

Web page(document) contains: object, such as an HTML file, image, addressable by an URL

stateless protocol 不记录client的信息

#### architecture: client-server

1. a client program/Web browsers(IE, ME, Firefox，浏览器是客户端，是Web的组成之一)
2. a server program/Web servers（管理并导向Web objects，web server always on with a fixed IP address）

#### transport protocol: TCP

communication: client-TCP socket of client side-TCP socket of server side-server

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302071455911.png" alt="image-20230207145508726" style="zoom: 50%;" />

### 2.2 Non-Persistent and Persistent Connections

*Conclusion：在client与server之间有两种TCP连接方式，一是非持久连接，另一种是持久连接，按字面意思，非持久连接是指每次请求响应都需要建立新的TCP连接，而持久连接则沿用同一个连接进行通信。每次连接需要三次握手，耗费近2个RTT（一个小数据包从客户端到服务器再回到客户端所花费的时间），其具体过程为，client发送TCP段请求连接（initiate）、server确认并响应TCP段，client确认并发送request message，此后server才将其需要的信息及索引发送给client。使用非持久连接的缺点很显然，内存（空间，每个新建的TCP连接参数都会占用空间）和延迟（时间，每次都需要至少2个RTT）。举例，假如client需要请求的web page有10个objects，使用非持久连接一共需要建立11个TCP连接。*

#### Non-Persistent connections

非持久连接意味着每次请求、响应都需要建立新的连接。步骤如下：

1. client初始化一个TCP connection在[server IP:80](80是HTTP的默认端口号）
2. client通过该TCP socket发送HTTP请求信息，包含页面路由信息
3. server接收请求信息并响应其需求，通过TCP socket发送该web page需要的objects（先发送其引用，即这些objects的路由）
4. server告知TCP可以关闭连接，但直到TCP确认client完整接收响应信息才会关闭。
5. 接下来每个objects都会重复1-3步

这意味着，假如有10个objects，一共需要生成11个TCP连接（最初的是web page请求，后续的是objects请求）。

#### round-trip time(RTT)

RTT: 一个小数据包从客户端到服务器再回到客户端所花费的时间。

3次握手：

1. client发送TCP段
2. server确认并响应一个TCP段
3. client确认（并发送request message）

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302071536749.png" alt="image-20230207153651596" style="zoom: 50%;" />



#### shortcomings

1. 内存：每次创建新TCP connections需要为TCP variables分配内存，对于同时处理多clients请求的Web server来说是巨大的负担
2. 延迟：每次都需要nearly 2 RTTs，一次建立连接，一次request & response

#### persistent connections

为了解决这些shortages，引出persistent connections，即部署在同一个Web Server上对多个Web Page的请求响应可以在同一个TCP connection中进行。HTTP/2能允许按优先级排列同一个TCP connection中的请求与响应。



### 2.3 HTTP Message Format

*Conclusion：主要介绍了HTTP message的格式，分为request message和response message。*

#### Request

```http
GET /somedir/page.html HTTP/1.1
Host: www.someschool.edu
Connection: close
User-agent: Mozilla/5.0
Accept-language: fr

// Entity body
```

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302071624958.png" alt="image-20230207162423825" style="zoom:50%;" />

**request line:** 

method

1. GET, mostly表单提交会用到GET而非POST，如`www.somesite.com/search?a&b`
2. POST, 此时request message的Entity body不为空，会携带client输入的信息
3. HEAD, 与GET类似，但只有response message，没有requested object，一般用于debug
4. PUT, 用于upload an object on a Web server，一般用于Web发布工具一起使用
5. DELETE，用于delete an object on a Web server

**header lines:**

1. Host, needed by the web proxy caches
2. Connection, non-persistent/persistent
3. User-agent, what browser client uses
4. Accept-language, preference in languages
5. etc.



#### Response Message

```http
HTTP/1.1 200 OK
Connection: close
Date: Tue, 18 Aug 2015 15:44:04 GMT
Server: Apache/2.2.3 (CentOS)
Last-Modified: Tue, 18 Aug 2015 15:11:03 GMT
Content-Length: 6821
Content-Type: text/html

(data data data data data ...)
```

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302071633678.png" alt="image-20230207163342537" style="zoom:50%;" />

**status line:**

1. protocol version field
2. status code
   1. 200 OK
   2. 301 Moved Permanently: requested **object**被永久移动，新的URL会在Location(one of the headers)中指出
   3. 400 Bad Request: incomprehensible request
   4. 404 Not Found: web page不存在于当前server
   5. 505 HTTP Version Not Supported: the requested protocol is not supported
3. corresponding status message

**header lines:**

1. Connection: close->发完就删
2. Date，这是server检索所需object并封装发送的时间
3. server，generated by what web server
4. Last-Modified，object最后修改日期
5. Content-Length
6. Content-Type

**enity body:** requested data

### 2.4 User-Server Interaction: Cookies

*Conclusion：用cookies来记录用户信息及其在browser上的行为，弥补HTTP无状态特点的不足，在这里client是browser，因此建立的是browser-cookie的键值关系。创建TCP连接时，server确认连接请求时给予当前client一个unique token作为cookie，server会将该cookie-client的对应关系记录在其数据库中，而client收到该cookie后会记录在本地浏览器缓存中，每次发送请求的时候会包含cookie信息。*

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302071710917.png" alt="image-20230207171033762" style="zoom:50%;" />

stateless与cookies的区别：stateless是指不记录client的状态，如果是已经response的完全一致的请求，还会再次response。cookies是记录client的历史数据信息，从该user-agent发出，在该web page进行了什么行为。

### 2.5 Web Caching

*Conclusion：简单来说就是在原服务器与client之间添加一个代理服务器，作为web缓存，客户端先向代理服务器发送请求，确认缓存中没有需求信息后，再由代理服务器向原服务器发送请求。好处是加速、减少流量，通常client-proxy server之间的带宽限制相比client-original server的更宽松，对于ISP企业网络，能够减少当前机构局域网对互联网的访问流量，减少整个互联网上的流量。用conditional GET可以保证cache和server之间的数据一致性。*

Web Cache-proxy server，在与client通信时是server，在与original server通信时是client，通常被ISP部署（用于大型机构网络，校园网或企业网）

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302071732195.png" alt="image-20230207173204070" style="zoom:50%;" />

#### advantages

1. time cost↓，cache与client之间的传输速度上限可能比cache与server之间的传输速度上限大
2. Web缓存可以大大减少机构到Internet的访问链路上的流量
3. 减少整个互联网流量

#### Content Distribution Networks(CDNs)

CDN公司在整个互联网上安装许多地理分布的缓存，从而本地化大部分流量。

#### Conditional GET

作用：用于判断cache中的object是否up to date，保证数据一致性（consistency between cache and server）。

运行原理：client向cache请求object，假如其不在cache中，cache向server请求，此时server会将object的last modified信息一起发送，cache保存这个信息。若在cache中，cache会先向server发送一个conditional GET作为一个up-to-date的确认，该报文包含If-modified-since属性，值与last modified相同，假如没有改变，server会response with`304 Not Modified`，不会再次传送object（entity body为空）。此时cache确认后，可以直接将缓存中的object响应给client。

## 3. Electronic Mail

### 3.1 Internet mail system

*Conclusion: 互联网邮件系统主要包含用户代理、邮件服务器与SMTP三个组成部分，用户代理对邮件进行操作，服务器储存、传送邮件，SMTP为服务器之间的TCP连接传输遵循的协议。*

1. user agents, where the user operates
2. mail servers, both client and server among other mail servers，存储数据、传送数据
3. SMTP(Simple Mail Transfer Protocol)，mail servers之间的TCP连接通信遵循SMTP服务

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302101655321.png" alt="image-20230210165522194" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302121554855.png" alt="image-20230212155423722" style="zoom: 67%;" />

### 3.2 Comparisons between HTTP & SMTP

|                        | HTTP                                                         | SMTP                                                         |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **pull/push**          | pull protocol，侧重download被upload在web server上的文件，TCP连接由client初始化（发出请求） | push protocol，侧重推送文件，TCP连接由发送邮件的mail server初始化 |
| **format limitations** | 无                                                           | 信息格式limited，7-bit ASCII format                          |
| **objects-messages**   | 将单个requested object封装在一条message，一对一              | 将所有objects封装在一条message                               |

### 3.3 Mail Message Formats

```
From: alice@crepes.fr
To: bob@hamburger.edu
Subject: Searching for the meaning of life.
```

```
telnet serverName 25
```

25 is the port number of SMTP server

### 3.4 Mail Access Protocols

*Conclusion: 收信人不能再采用SMTP来接收message，因为SMTP是push protocol而非pull protocol，因此要采用mail access protocols: POP3, IMAP, HTTP。POP3的运行具有3阶段，认证（用户名密码登陆）、业务（撤回、标记删除邮件、保存邮件数据）、更新（根据标记删除邮件，quit结束会话）。IMAP相比POP3，能够保存会话中的用户状态信息，具有文件夹分类功能，这也导致其实现比POP3要复杂得多。*



#### Post Office Protocol—Version 3 (POP3)

3 phases:

1. authorization: username-password
2. transaction: retrieve/mark for deletion/obtain data
3. update: issue `quit` command, end the session; delete the marked messages

commands: `list`,`retr`,`dele`,`quit`

#### Internet Mail Access Protocol (IMAP)

相比POP3，多了文件夹分类功能，这意味着IMAP会保存IMAP session中的用户状态信息，例如分类和文件夹名

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302122139584.png" alt="image-20230212164315736" style="zoom: 67%;" />





## 4. DNS(Domain Name System)

已知：用hostname和IP地址可以定位一个主机

需求：路由器需要定长的、分级的IP地址，需要建立hostname到IP地址的映射，即域名解析DNS

### 4.1 Services provided by DNS

*Conclusion: DNS包括在DNS servers层次结构中实现的分布式数据库，以及允许host查询分布式数据库的application-layer protocol。DNS是一个应用，其提供的服务为伪域名查询和负载平衡。*

#### definition

1. 在DNS服务器层次结构中实现的分布式数据库
2. 允许主机查询分布式数据库的应用层协议

一般是UNIX机器，DNS protocol一般在UDP和端口53上运行，一般被其他应用层协议，如HTTP,SMTP应用。

**应用原理：**

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302140124680.png" alt="image-20230214012429591" style="zoom:67%;" />

1. browser client将hostname传送给DNS应用的client side
2. DNS client将hostname传送给server，等待server回应该hostname对应的IP地址
3. browser client接收DNS应用传来的IP地址，初始化TCP连接

**services:**

1. host/mail server aliasing，主机/邮箱伪域名，DNS能找到一个alias hostname的canonical hostname及其IP
2. Load Distribution，负载平衡，对于重复的不同终端上的web servers，其canonical hostname一致，但拥有不同的IP地址，DNS能获取并响应这组IP地址，但会对其顺序进行轮换，由于client一般会按顺序优先采用这个地址集中的IP地址，从而起到了分流的作用。

### 4.2 How DNS Works

*Conclusion: 中心化设计的DNS具有单点故障（一损俱损）、流量代价、中心数据库距离远、维持成本高（经常需要更新hosts信息）的缺点，因此DNS服务器采用分布式和分级设计，具有三层分级，root，TLD，authoritative，每一层分别提供下一层的IP地址，最后还有local DNS，作为host到三级DNS服务器的proxy server，也能储存TLD server和短期内查询过的hostname的缓存。*

shortages of a centralized design of DNS:

1. single point of failure 单点故障
2. traffic volume 流量代价
3. distant centralized database 遥远
4. maintenance 维持成本（更新hosts信息）

因此要有distributed and hierarchical design

#### 3 classes of DNS servers

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302122139452.png" alt="image-20230212210955098" style="zoom: 80%;" />

1. Root DNS servers, 提供TLD servers的IP地址
2. Top-level domain(TLD) servers, `com, org, net, edu, and gov`，提供authoritative DNS servers的IP地址
3. Authoritative DNS servers, 在Internet上拥有可公开访问的主机(如Web服务器和邮件服务器)的每个**组织**都必须提供可公开访问的DNS记录，这些记录包含hostname到IP地址的映射。组织的授权DNS服务器保存这些DNS记录。
4. local DNS server, 严格来说不属于DNS服务器层级，但是DNS架构的中心；每个ISP都会有一个local DNS server，作为一个在请求host和DNS server hierarchy（以上三种）之间的proxy server

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302122139550.png" alt="image-20230212213909456" style="zoom:67%;" />

从请求主机到本地DNS服务器的查询是递归的，其余查询是迭代的。

#### cache

local DNS server could **cache** IP of

1. TLD servers
2. same hostnames recently

### 4.3 DNS Records and Messages

Resource records(RRs)：cache中对hostname-IP mappings的记录

```
(Name, Value, Type, TTL)
```

#### 参数

| Type  | Name           | Value                                                        | Example                                  |
| ----- | -------------- | ------------------------------------------------------------ | ---------------------------------------- |
| A     | hostname       | IP address                                                   | $(relay1.bar.foo.com, 145.37.93.126, A)$ |
| NS    | domain         | 能提供该域名族host的IP地址的authoritative DNS server的hostname | $(foo.com,dns.foo.com, NS) $             |
| CNAME | alias hostname | canonical hostname                                           | $(foo.com,relay1.bar.foo.com, CNAME)$    |
| MX    | alias hostname | canonical name of a mail server                              | $(foo.com, mail.bar.foo.com, MX)$        |

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302131605629.png" alt="image-20230213160553467" style="zoom: 80%;" />

#### 安全性

针对DNS的攻击：

1. **DDoS（分布式拒绝服务）**带宽泛洪攻击，向每个root DNS server发送大量packet，使得大多数合法DNS请求得不到回答。
2. 截获来自主机的请求，并返回伪造的回答，替换服务器的缓存信息，能将Web用户重定向到病毒Web站点。

### 4.4 Distribution

#### P2P File Distribution

从单一服务器向大量主机分发一个大文件。将不同小的片段发送给不同主机，让主机之间相互匹配传输缺失的部分；比client-server更高效。eg. BitTorrent

#### DASH

DASH(Dynamic Adaptive Streaming over HTTP) 经HTTP的动态适应性流，改变视频流编码，360p-1080p

#### CDN (Content Distribution Network)

CDN，内容分发网在为client播放流媒体时缓存该视频，当缓存满了则移除较少访问的视频。

**分布方式：**

1. enter-deep：在ISPs附近布置服务器，这样服务器的数量会比较大，Akamai采用这种方法
2. bring-home：在Internet Exchange Points（IXPs）布置服务器，这样服务器数量较少，Limelight等CDN公司采用这种方法

**运行原理：**

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302131734365.png" alt="image-20230213173431201" style="zoom:50%;" />

#### cluster selection strategies

1. geographically closest 地理最近，弊端是每次都固定一个cluster，不考虑现实延迟和宽带上限，有时地理最近在网络上不是最近，且有些终端与定位较远的LDNSs绑定
2. real-time measurements 实时测量，让CDN中的clusters周期性地测量到达速度最快的LDNSs

### 4.5 Netflix, YouTube, and kankan

#### Netflix

网飞的web pages分发由Akamai公司运营，视频内容分发主要依靠Amazon cloud & private CDN。该网站是在亚马逊云中的亚马逊服务器运营的，亚马逊云实现了以下功能：

1. Content ingestion，把原片上传到亚马逊云中的hosts
2. Content processing，亚马逊云对视频进行各种编码并存储，以适应不同终端、流畅度、清晰度播放（DASH）
3. Uploading versions to its CDN，将这些不同版本的视频上传到其private CDN

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302140109437.png" alt="image-20230214010905870" style="zoom:50%;" />

push caching:

Netflix CDN uses push caching rather than pull caching，意思是在非高峰期定期将内容推送到服务器，而非高峰期需要的时候再pull

#### YouTube

private CDN/pull caching/cluster-selection：找出RTT最短的cluster，但为了保持平衡一般会通过DNS定向到更远的cluster/*现在已经有了适应性流媒，有一些视频使用DASH（书是2017出版，2018 stackoverflow说youtube还在使用dash）*

### 4.6 client-server application using UDP/TCP

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302131813791.png" alt="image-20230213181307672" style="zoom: 67%;" />



<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302140031194.png" alt="image-20230214003116046" style="zoom:67%;" />







*记点好玩的：*

*RFC 2324: “[超文本咖啡壶控制协议](https://baike.baidu.com/item/超文本咖啡壶控制协议/8410606?fromModule=lemma_inlink)”（Hyper Text Coffee Pot Control Protocol，乍有其事的写了HTCPCP这样看起来很专业的术语缩写字）。以及如前面所提到纪念RFC的30周年庆的RFC文件。*





















































































predominant，相比dominant有evident/obvious的意思

jargon 行话，黄皓石

analog[c.] 类比 analogous/similar

invoke[v.] 援引；祈求；唤起 invoke...as

glitch[c.] 小故障；失灵；[电子]短时脉冲波干扰

impairment [c.] （身体或智力方面的）损伤，缺陷，障碍

fluctuate[v.] 波动，摇摆 +with 随着……摆动 wave

confidentiality[n.] 保密性

onslaught[c.] 攻击，猛攻；（常指难以应付的）大批，大量；猛烈抨击

welfare[u./adj.]n. 幸福，安康；福利救济，社会福利；（给失业者和穷人的）福利救济金 adj. 福利的；接受社会救济的 welfare services

no-frills([c.])[adj.] simple, basic; no-frills service

realm[c.] （知识、活动、思想的）领域，范围；<正式>王国；（动物学）界（对地球表面所作的最主要的生物地理划分）

delineate[v.] demonstrate; illustrate（详细地）描述，解释；标明，标示（边界）

elaborate

- adj. 复杂的，详尽的；精心制作的
- v. 详细说明，详尽阐述；精心制作

semantics[u.] 语义学

conspicuously adv. 显著地，明显地；超群地，惹人注目地

terminology n. （某学科的）术语；有特别含义的用语，专门用语；术语学

back-of-the-envelope calculation 粗略的计算

propagation n. （动植物等的）繁殖，增殖；（观点、理论等的）传播；（运动、光线、声音等的）传送

perceive  v. 感知；认为；领会

stale (out of date)

- adj. （食物）不新鲜的，变味的；气味不清新的，难闻的；没有新意的，老掉牙的；（因长期从事同一工作而）疲沓的，厌倦的；（支票，合法要求）（因过期而）失效的
- v. （使）不新鲜，（使）走味，（使）陈旧；（动物，尤指马）撒尿

infancy [u.] in one's infancy 婴儿期，未成年

utilize [v.]use，apply to

ubiquity [u.] 无处不在，one's ubiquity in...

archaic [adj.]adj. （词，语言风格）古体的，已不通用的；过时的，陈旧的；古代的，早期的; ancient;stale but in word descriptions

allude vi. 暗示，暗指；略微提及；（某艺术家或艺术品）让人想起（从前的作品或风格）

tacitly adv. 肃静地；沉默地；心照不宣地

phase ;/stage n. （发展或变化的）阶段，时期；做出某种行为的短时期；（动物生活周期或年周期的）阶段，期；（月亮的）位相，盈亏；同步，协调；（化）相；（动）（由遗传或季节引起的）动物颜色变化（期）；（物理）相位，相角；（线圈的）匝，（多相电机或电路的）连接；（语言学）（系统语法用语）相（指链接动词与后续动词的关系）；相结构（含有链接动词和后续动词的结构）

nomadic adj. 游牧的；流浪的

paradigm n. 典范，范例；样板，范式；词形变化表；纵聚合关系语言项

reconcile v. 调和，使协调一致；（使）和解，（使）恢复友好关系；调停，调解（争吵）；使顺从于，使接受；核对，查核（账目）

canonical/authentic/standard adj.权威的，标准的



