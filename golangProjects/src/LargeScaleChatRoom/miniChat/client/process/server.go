package process

import (
	"bufio"
	"fmt"
	"os"
)

/**
1. show model status
2. keep Dial connecting
*/

func ShowMenu(userName string) {
	fmt.Printf("----- User %v -----\n", userName)
	fmt.Println("----- 1 Online users list -----")
	fmt.Println("----- 2 send message -----")
	fmt.Println("----- 3 messages list -----")
	fmt.Println("----- 4 exit -----")
	var key int
	smsProc := &SmsProcess{}
	fmt.Scanf("%d\n", &key)
	switch key {
	case 1:
		outputUserStatus()
	case 2:
		fmt.Println("Enter a message: ")
		var content string
		reader := bufio.NewReader(os.Stdin)
		content, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println("read message error, please tray again")
		}
		err = smsProc.SendGroupMessage(content)
		if err != nil {
			fmt.Println("send message failed")
		}
	case 3:
		fmt.Println()
	case 4:
		fmt.Println("exit")
		return
	default:
		fmt.Println("undefined opt...")
	}
}
