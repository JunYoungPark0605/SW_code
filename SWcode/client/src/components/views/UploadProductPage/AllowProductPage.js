import React, { useState } from 'react'
import { Form, Input, Icon, Switch } from 'antd';
import FileUpload from '../../utils/FileUpload';
import Axios from 'axios';
const { TextArea } = Input;

switch(categories){
    case 1 : const DetailCategories = [
        { key: 1, value: "국밥" },
        { key: 2, value: "찌개" },
        { key: 3, value: "전골" },
        { key: 4, value: "닭볶음탕" },
        { key: 5, value: "정식" },
        { key: 6, value: "탕" },
        { key: 7, value: "전" },
        { key: 8, value: "덮밥" }
    ]
    break;
    case 2 : const DetailCategories = [
        { key: 1, value: "게장" },
        { key: 2, value: "조개구이" },
        { key: 3, value: "해물탕" },
        { key: 4, value: "회" },
        { key: 5, value: "생선구이" },
        { key: 6, value: "찜" },
        { key: 7, value: "갑각류" }
    ]
    break;
    case 3 : const DetailCategories = [
        { key: 1, value: "돼지고기" },
        { key: 2, value: "소고기" },
        { key: 3, value: "뒷고기" },
        { key: 4, value: "막창/대창/곱창" },
        { key: 5, value: "양고기" },
        { key: 6, value: "닭갈비" },
        { key: 7, value: "오리고기" },
        { key: 8, value: "육회" }
    ]
    break;
    case 4 : const DetailCategories = [
        { key: 1, value: "족발" },
        { key: 2, value: "보쌈" }
    ]
    break;
    case 5 : const DetailCategories = [
        { key: 1, value: "돈까스" },
        { key: 2, value: "스시" },
        { key: 3, value: "텐동" },
        { key: 4, value: "오꼬노미야끼" }
    ]
    break;
    case 6 : const DetailCategories = [
        { key: 1, value: "파스타" },
        { key: 2, value: "스테이크" }
    ]
    break;
    case 7 : const DetailCategories = [
        { key: 1, value: "마라탕" },
        { key: 2, value: "중화요리" },
        { key: 3, value: "전통중국요리" }
    ]
    break;
    case 8 : const DetailCategories = [
        { key: 1, value: "떡볶이" },
        { key: 2, value: "김밥" },
        { key: 3, value: "핫도그" }
    ]
    break;
    case 9 : const DetailCategories = [
        { key: 1, value: "냉면" },
        { key: 2, value: "밀면" },
        { key: 3, value: "칼국수" },
        { key: 4, value: "쌀국수" },
        { key: 5, value: "잔치국수" },
        { key: 6, value: "수제비" },
        { key: 7, value: "모밀" },
        { key: 8, value: "야끼소바" },
        { key: 7, value: "라면/라멘" },
        { key: 8, value: "우동" }
    ]
    break;
    case 10 : const DetailCategories = [
        { key: 1, value: "치킨" }
    ]
    break;
    case 11 : const DetailCategories = [
        { key: 1, value: "햄버거" }
    ]
    break;
    case 12 : const DetailCategories = [
        { key: 1, value: "피자" }
    ]
    break;
    case 13 : const DetailCategories = [
        { key: 1, value: "토스트" },
        { key: 2, value: "샌드위치" },
        { key: 3, value: "샐러드" }
    ]
    break;
    case 14 : const DetailCategories = [
        { key: 1, value: "도시락/죽" }
    ]
    break;
    case 15 : const DetailCategories = [
        { key: 1, value: "디저트" }
    ]
    break;
}

const Spots = [
    { key: 1, value: "학교 앞" },
    { key: 2, value: "가로수길" }
]

function AllowProductPage(props) {

    const [Title, setTitle] = useState("")
    const [Description, setDescription] = useState("")
    const [DetailCategories, setDetailCategories] = useState(1)
    const [Spots, setSpots] = useState(1)
    const [Images, setImages] = useState([])
    const [CallNumber, setCallNumber] = useState("");
    const [reviewlink, setreviewlink] = useState("");

    const titleChangeHandler = (event) => {
        setTitle(event.currentTarget.value)
    }
    
    const handleChangeCallNumber = (event) => {
        console.log(event.currentTarget.value)

        setCallNumber(event.currentTarget.value)
    }

    const handleChangereviewlink = (event) => {
        console.log(event.currentTarget.value)

        setreviewlink(event.currentTarget.value)
    }

    const descriptionChangeHandler = (event) => {
        setDescription(event.currentTarget.value)
    }


    const DetailCategoriesChangeHandler = (event) => {
        setDetailCategories(event.currentTarget.value)
    }

    const spotsChangeHandler = (event) => {
        setSpots(event.currentTarget.value)
    }

    const updateImages = (newImages) => {
        setImages(newImages)
    }

    const submitHandler = (event) => {
        event.preventDefault();

        if (!Title || !Description || !DetailCategories || !Spots || Images.length === 0) {
            return alert(" 모든 값을 넣어주셔야 합니다.")
        }


        //서버에 채운 값들을 request로 보낸다.

        const body = {
            //로그인 된 사람의 ID 
            writer: props.user.userData._id,
            title: Title,
            description: Description,
            callNumber: CallNumber,
            reviewlink: reviewlink,
            images: Images,
            deTailCategories: DetailCategories,
            spots: Spots
        }

        Axios.post('/api/product', body)
            .then(response => {
                if (response.data.success) {
                    alert('상품 업로드에 성공 했습니다.')
                    props.history.push('/')
                } else {
                    alert('상품 업로드에 실패 했습니다.')
                }
            })
    }


    return (
        <div style={{ maxWidth: '700px', margin: '2rem auto' }}>
            <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                <h2> 음식점 업로드</h2>
            </div>

            <Form onSubmit={submitHandler}>
                {/* DropZone */}
                <FileUpload refreshFunction={updateImages} />

                <br />
                <br />
                <label>이름</label>
                <Input onChange={titleChangeHandler} value={Title} />
                <Icon type="exclamation-circle" style={{ color: 'rgb(256,50,50)' }} />
                
                <label style={{ color: 'rgb(256,50,50)' }}>중복되는 음식점이 있는지 확인 먼저 부탁드립니다.</label>
                
                <br />
                <br />
                <label>음식점 설명</label>
                <TextArea onChange={descriptionChangeHandler} value={Description} />
                <br />
                <br />
                <label>전화번호</label>
                <Input
                    onChange={handleChangeCallNumber}
                    value={CallNumber}
                />
                <br />
                <br />
                <label>리뷰 링크</label>
                <Input
                    onChange={handleChangereviewlink}
                    value={reviewlink}
                />
                <br />
                <br />
                <select onChange={DetailCategoriesChangeHandler} value={DetailCategories}>
                    {DetailCategories.map(item => (
                        <option key={item.key} value={item.key}> {item.value}</option>
                    ))}
                </select>
                <br />
                <br />
                <label>위치</label>
                <br />
                <br />
                <button type="submit">
                    확인
                </button>
            </Form>


        </div>
    )
}

export default AllowProductPage
