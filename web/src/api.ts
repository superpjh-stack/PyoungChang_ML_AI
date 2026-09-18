export async function api<T>(path:string, init?:RequestInit):Promise<T>{
  const response=await fetch(path,{...init,headers:{'Content-Type':'application/json',...(init?.headers||{})}});
  if(!response.ok){const body=await response.json().catch(()=>({detail:'요청에 실패했습니다.'}));throw new Error(typeof body.detail==='string'?body.detail:'입력값을 확인해 주세요.');}
  return response.json();
}
