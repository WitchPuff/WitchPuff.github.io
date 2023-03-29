---
layout: post
title:  "Computer Network: Data Plane of Network Layer"
date:   2023-2-10 01:36:30
categories: Courses
tag: Computer Network

---
* TOC
{:toc}




# Network Layer: Data Plane



## 1. Network Layer Services

1. forwarding，data plane, hardware
2. routing，control plane，routing algorithm, software, according to forwarding table(packet header-output matching)

SDN(Software-Defined Networking)，软件定义网络

services:

1. Guaranteed delivery，确保交付
2. Guaranteed delivery with bounded delay，在一定时延范围内确保交付
3. In-order packet delivery，有序交付
4. Guaranteed minimal bandwidth，最小带宽确保交付
5. Security



## 2. Router

![image-20230220145842781](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302201458056.png)

### 2.1 input port processing

1. match: lookup, longest prefix-matching rule
2. action: ->switching fabric(queued here, then->the specific output port->output link)

存储转发表的copy

### 2.2 Switching fabric

#### 2.2.1 switching via memory

packets需要先被存储在内存中，再转发。obviously limited by the memory bandwidth，路由器的总转发量（B/2）不超过内存带宽（B）的一半，为了防止读写冲突、防止输出端口阻塞。
![image-20230220155220847](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302201552952.png)

#### 2.2.2 Switching via a bus

输入端口为packet标记其输入输出端口信息，一次只有一个数据包在一根shared bus上，每个output port都会接收到该数据包，但只有匹配的output port会保留它，此后移除掉标签信息。obviously limited by the bus speed
![image-20230220160302825](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302201603910.png)

#### 2.2.3 Switching via an interconnection network

解决了bus speed limitations，non-blocking（因为有多条总线，但单条总线上一次还是只能传输一个packet），对于N input port to N output port，共有2N buses
![image-20230220160531473](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302201605567.png)



### 2.3 Output Port Processing

#### 2.3.1 Queuing

1. Input Queuing，Head-Of-the-Line, HOL, 线路前部阻塞
   ![image-20230220211428224](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302202114308.png)
2. Output Queuing，多个packet同时到达输出端口，此时需要排队等待传输到output link，当缓存区满需要丢包，移除新到达的（drop-tail） or移除已经在排队的分组。
   ![image-20230220211340652](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302202113753.png)

#### 2.3.2 Packet Scheduling

on the output port

1. FIFO/FCFS
2. Priority Queuing，高优先级队列优先
   1. non-preemptive，非抢占，不截断当前packet的传输
   2. preemptive
3. RR/WFQ(Weighted Fair Queuing)
   1. RR: multiple classes轮换，根据work-conserving queuing，只要当前类别队列为空，就立刻轮换到下一个非空类别队列
   2. WFQ：与RR相比，为每个class分配了权重，w与每个类在任何时间内可能收到的不同数量的packet有关。

## 3. Internet Protocol(IP)





<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302202304808.png" alt="image-20230220230435700" style="zoom:67%;" />



fragment: 被MTU限制，IP datagram中的data可能需要被切割成fragments（片），但在到达transport layer前进行reassemble。切割和重组一般都在网络层的路由器中进行。如何组装：用header中的第二行，来自同一个datagram的（identifier，源和目的IP地址）一致（三元组标识）；最后一个fragment的flags=0，其余设为1；offset用于指定其在原datagram中的位置，检查是否有片段遗失。



interface: between host and link; between a link and a router; 

在Internet中，每台主机接口、路由器接口都要拥有全球唯一的IP地址

点分十进制记法

Internet地址分配策略：无类别域间路由选择（Classless Interdomain Routing，CIDR），一般化子网寻址

network mask 子网掩码表示方法：a.b.c.d/x(223.1.1.0/24)， x为该子网prefix长度，转发时只需要考虑前x bits

<img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302202320401.png" alt="image-20230220232041262" style="zoom:67%;" />

address aggregation/route aggregation/route summarization: ISP向外界通告“send me目的IP为所在a.b.c.d/x（包括多个子网）的所有datagram”。有点像聚合多个子网，找到更短的共同prefix。路由选择时采用最长前缀匹配，

classful addressing分类编址：A-8 bits，B-16bits，C-24bits

broacast：255.255.255.255，报文交付给同一网络中的所有主机

DHCP(Dynamic Host Configuration Protocol)，动态主机配置协议，自动将一个主机接入网络，即插即用/零配置，即网络管理员不必手动配置；client-server，server给新到达网络的client分配IP地址

1. client发送DHCP discover message in UDP segment，用广播机制，让DHCP 发现自己
2. DHCP server收到DHCP discover message，发送DHCP offer message作为回应，也用广播机制，便于client找到最近的server，包含推荐的客户IP地址、子网掩码和地址租用期（该分配的IP地址是有期限的）
3. client选择server，向其发送DHCP request message
4. server发送DHCP ACK message进行确认，client收到后交互完成



Network Address Translation，NAT，网络地址转换

用于专用网络或具有专用地址的地域

NAT后的主机发送datagram，经过NAT，NAT路由器重写datagram的目的IP与port，再发出，收到响应时根据NAT转换表传送给真正的目的主机

NAT穿越（NAT traversal）



IPv6

现实原因：IPv4的32位地址快用完了

![image-20230221084716816](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302210847953.png)

1. fragments：不允许在intermediate上切割/重组，只能在source和destination中进行，如果超出链路限制，路由器会直接drop并发送ICMP报错
2. checksum：既然transport和link layer都提供我们网络层就不提供了，否则每一跳都要计算checksum代价很大，这是ipv4的弊端
3. options：放在next header field中

IPv4->IPv6

![image-20230221090122161](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202302210901311.png)

流表

matching+action

actions

1. forwarding，Forward(路由器的interface)
2. dropping
3. modify-field

functions：

1. simple forwarding
2. 负载均衡（分流）
3. firewall（只匹配转发特定地址的流量）





monolithically/monolithic 武断的

euphemism 委婉说法
