---
layout: post
title:  "Computer Network: Control Plane of Network Layer"
date:   2023-2-10 01:36:30
categories: Courses
tag: Computer Network

---
* TOC
{:toc}


# Network Layer: Control Plane

路由算法

1. 集中式：全局网络知识计算，需要知道网络中每条链路的开销，具有全局状态信息的算法被称为链路状态
2. 分散式：迭代、分布式计算最低开销路径

静态/动态

负载敏感/负载迟钝



链路状态路由选择算法（Link state broadcast，LS算法），知道每条链路的开销

1. Dijkstra
2. prim

距离向量路由选择算法（Distance-Vector，DV），迭代异步分布式

$d_x(y)=min_v\{c(x,v)+d_v(y)\}$

链路开销改变与链路故障，只有更新最短链路开销时才广播，更新失效的链路

增加毒性逆转，假设xyz三角，z-y-x最短，z向y发送的距离表将欺骗其z无法到达x，这样y就不会去尝试到达x，解决特定环路问题

LS缺陷：全局泛洪，代价过大

DV缺陷：无法收敛，无穷迭代

### RIP, Router Information Protocol

基于DV算法

周期性/请求，向邻居router节点发送距离矢量信息，更新到不同host节点的最短路径

在网络层以进程方式实现，借助UDP协议传输距离矢量报文

### OSPF(Open Shortest Path First，开放式最短路径优先)

AS, Autonomous System，自治系统

Link State Protocol

自治系统内部路由选择协议，向自治系统内所有其他路由器**周期性（或发生变化时）广播**更新路由选择信息，通过全网泛洪

在IP数据包上直接传送，不需要用到传输层

允许多个代价相同的路由，用不同方式计算代价，跳数/延迟

层次化网络，每个link state packet可以仅在area内泛红，而不会波及到较大规模的全网

1. 对相同AS非相同area内的通信，需要路由到backbone（骨干区域）再中转到对应目标
2. 对不同AS，则路由到边界border路由器



### 层次化路由

规模性（代价）/管理性（可扩展性）

1. 自治区内采用合适的路由选择协议和算法，内部网关协议
2. 自治区间，BGP

BGP, Border Gateway Protocol, 边界网关协议，基于改进的DV算法，会告知到达子网的详细AS路径，避免环路，加快收敛

AS间，eBGP

AS内，iBGP，毫无隐瞒，最大性能

网关路由器运行eBGP and iBGP

收集子网内的路由可达信息（iBGP），通过TCP segment传达给另一个自治区的网关路由器（eBGP）

1. AS-path：前缀通告所经过的AS及代价（AS内部代价和，注意AS视作一个点 ）
2. next hop：链路节点，热土豆路由，谁最近选谁

策略：为路径打分，代价/安全性；接收/过滤，通告/隐瞒

SDN

match-action

北向接口：操作网络应用

南向接口：转发流表