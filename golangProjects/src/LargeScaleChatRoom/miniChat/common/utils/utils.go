package utils

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net"
	"redisLearn/miniChat/common/Message"
)

type Transfer struct {
	Conn   net.Conn
	Buffer [8096]byte // buffer for messages
}

func (trans *Transfer) WritePackage(data []byte) (err error) {
	MesLen := len(data)
	binary.BigEndian.PutUint32(trans.Buffer[0:4], uint32(MesLen))
	n, err := trans.Conn.Write(trans.Buffer[0:4])
	if n != 4 || err != nil {
		fmt.Println("MesLen Write err: ", err)
	}
	// send data
	n, err = trans.Conn.Write(data[0:MesLen])
	if n != MesLen || err != nil {
		fmt.Println("Write Message err: ", err)
	}
	return
}

func (trans *Transfer) ReadPackage() (mes Message.Message, err error) {

	//conn没有被关闭的情况下才会阻塞
	n, err := trans.Conn.Read(trans.Buffer[:4]) //长度信息
	if n != 4 || err != nil {
		fmt.Println("trans.Conn.Read err: ", err, ", read len: ", n)
		return mes, err
	}
	var pkgLen uint32
	pkgLen = binary.BigEndian.Uint32(trans.Buffer[:4]) //长度
	//根据pkgLen读取内容
	n, err = trans.Conn.Read(trans.Buffer[:int(pkgLen)])
	if n != int(pkgLen) || err != nil {
		fmt.Println("trans.Conn.Read failed err=", err, "pkgLen: ", pkgLen)
		return mes, err
	}
	err = json.Unmarshal(trans.Buffer[:pkgLen], &mes)
	if err != nil {
		fmt.Println("Unmarshal message err: ", err)
		return mes, err
	}
	return mes, nil
}
