package model

import (
	"net"
	"redisLearn/miniChat/common/Message"
)

type CurUser struct {
	Conn net.Conn
	Message.User
}
