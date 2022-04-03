import React, { useState } from 'react'
import { Button, Input } from 'antd';
import axios from 'axios';
import { useSelector } from 'react-redux';
import SingleComment from './SingleComment';
import ReplyComment from './ReplyComment';
const { TextArea } = Input;

function Comments(props) {
    const user = useSelector(state => state.user)
    const [Comment, setComment] = useState("")

    const handleChange = (e) => {
        setComment(e.currentTarget.value)
    }

    const onSubmit = (e) => {
        e.preventDefault();

        const variables = {
            content: Comment,
            writer: user.userData._id,
            productId: props.productId
        }

        axios.post('/api/comment/saveComment', variables)
            .then(response => {
                if (response.data.success) {
                    setComment("")
                    props.refreshFunction(response.data.result)
                } else {
                    alert('Failed to save Comment')
                }
            })
    }


    if (user.userData && !user.userData.isAuth) {
        return (
            <div>
                <br />
                <p> 댓글</p>
                <hr />
                {console.log(props.CommentLists)}

                {props.CommentLists && props.CommentLists.map((comment, index) => (
                    (!comment.responseTo &&
                        <React.Fragment>
                            <SingleComment comment={comment} productId={props.productId} refreshFunction={props.refreshFunction} />
                            <ReplyComment CommentLists={props.CommentLists} productId={props.productId} parentCommentId={comment._id} refreshFunction={props.refreshFunction} />
                        </React.Fragment>
                    )
                ))}
            </div>
    
          
          
        
        )
      } else{
    return (
        <div>
            <br />
            <p> 댓글</p>
            <hr />
            {/* Comment Lists  */}
            {console.log(props.CommentLists)}

            {props.CommentLists && props.CommentLists.map((comment, index) => (
                (!comment.responseTo &&
                    <React.Fragment>
                        <SingleComment comment={comment} productId={props.productId} refreshFunction={props.refreshFunction} />
                        <ReplyComment CommentLists={props.CommentLists} productId={props.productId} parentCommentId={comment._id} refreshFunction={props.refreshFunction} />
                    </React.Fragment>
                )
            ))}



            {/* Root Comment Form */}
            <form style={{ display: 'flex' }} onSubmit={onSubmit}>
                <TextArea
                    style={{ width: '100%', borderRadius: '5px' }}
                    onChange={handleChange}
                    value={Comment}
                    placeholder="write some comments"
                />
                <br />
                <Button style={{ width: '20%', height: '52px' }} onClick={onSubmit}>등록</Button>
            </form>

        </div>
    )
}
}
export default Comments
