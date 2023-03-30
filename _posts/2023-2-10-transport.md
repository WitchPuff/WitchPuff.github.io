---
layout: post
title:  "Computer Network: Transport Layer(2)"
date:   2023-2-11 01:40:30
categories: Courses
tag: ["Computer Network"]
math: true
---

* TOC
{:toc}


# Transport Layer

## 1. Transport-Layer Services

Conclusion：相比运行在hosts（IP地址）之间的网络层协议，传输层协议运行在不同hosts的processes之间，实现logical communication，且其部署在end systems中而非routers中。网络层协议有UDP与TCP，两者都提供多路复用/解复用与错误检测的服务，但TCP还提供可靠数据传输、流控制、拥塞控制的服务。

服务：传输层协议提供了运行在不同hosts上运行的**processes**实现逻辑通信logical communication（指他们在逻辑上直接连接，但实际上可能经过了无数routers）。相对的，一个网络层协议在**hosts**之间实现逻辑通信。传输层协议能提供的服务被底层的网络层协议所限制。

部署：传输层协议在**end systems**中实现而非routers中，路由器只负责转发datagram。

行为：传输层协议将应用层message封装为传输层message，传输给网络层，网络层再封装成数据报datagram。也就是说每一层就都会将数据封装为能在该层传输、向下层传输的packet。

### 1.1 Transport layer protocols

between **processes** running on hosts

1. UDP(User Datagram Protocol)-datagram/segment

2. TCP(Transmission Control Protocol)-segment

两者都有/UDP仅有的services：

1. transport-layer multiplexing and demultiplexing 传输层的多路复用与解复用，将host-host delivery扩展到process-process delivery
2. integrity/error checking

TCP独有：

1. reliable data transfer
2. congestion control

### 1.2 Network Layer protocols

between **hosts(IP addresses)**

IP(Internet Protocol)-datagram

services:

1. best-effort delivery service

2. no guarantees, unreliable service



## 2. Multiplexing & Demultiplexing

Conclusion：在多主机、多应用、多进程、多socket上，不同应用间的通信是通过segment在socket中传输实现的，多路复用/解复用就用于定位一个segment的目的socket与源socket是什么，解复用是发送至socket，多路复用是从socket接收并发至网络层，这通过两者实现：1）唯一标识符socket，2）segment中指定源与目的socket **port number**。UDP socket只关注目的地，而TCP socket由于面向连接，会建立源-目的地的一对一连接。

已知：多主机，多应用，一个应用对应多进程，一个进程对应多socket（多线程），segment要通过socket传输。

需求：问题在于如何找到正确的socket，多路复用/解复用服务能解决这个问题。所有计算机网络都需要多路复用/解复用服务。

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302141743186.png" alt="image-20230214174355071" style="zoom:50%;" />

### 2.1 definition

1. demultiplexing，解复用，就是将transport-layer segment中的data传送到正确的socket中

2. multiplexing，多路复用，将从不同socket接收到的data chunks封装成segments，添加头信息（用于解复用），发送至network layer

要实现这项技术这需要：

1. unique identifiers for sockets 唯一标识符socket

2. segment中含有字段指定destination socket，而这包括source port number & destination port number（每个端口号都是16-bit number，传输层一般指定1024-65535 ，0-1023是著名端口号，一般为一些广泛使用的application保留，如HTTP-80）

   <img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302141750994.png" alt="image-20230214175055942" style="zoom:50%;" />

一般来说，应用的client side会让传输层自动生成socket端口号，而server side则指定一个特定端口号

### 2.2 Multiplexing & Demultiplexing in UDP

$return\ address = source\ port\ number + source\ IP\ address$

$UDP\ socket= (destination\ IP\ address,\ destination\ port\ number)$

UDP只关注目的地，一个UDP socket能接收source不同、发送destination不同的信息。

### 2.3 Multiplexing & Demultiplexing in TCP

$TCP\ socket=(source\ IP\ address,\ source\ port\ number,\ destination\ IP\ address,\
destination\ port\ number)$

TCP关注source & destination，完全双向一对一连接。

**原理：**

1. TCP server具有一个welcoming socket，用于等待连接请求
2. TCP client创建一个socket，通过其发送连接请求segment，会在其中指定其该四元组
3. TCP server收到连接请求后，根据该请求segment中的四元组创建一个新的socket

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302151702629.png" alt="image-20230215170204494" style="zoom:67%;" />

### 2.4 Security

攻击者可以监听/攻击一些知名应用的默认端口，造成其缓冲区溢出，可能可以在该主机上执行任何代码。因此有必要查询本机中监听端口的进程，可以用nmap。

