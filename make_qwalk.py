

####################################################
def read_trail_ecp(f):
  """Returns effective charge, r, and potential*r
  Units are in Rydbergs and Bohr
  """
  line=f.readline() #Comment
  line=f.readline() #Atomic number
  spl=f.readline().split()
  atnum=int(spl[0])
  effc=float(spl[1])
  print("atomic number",atnum,'effective charge',effc)
  line=f.readline()
  units=f.readline()
  if 'rydberg' not in units:
    print("Don't support units of",units)
    quit()

  line=f.readline()
  nloc=int(f.readline().split()[0])
  f.readline() #NRULE override
  f.readline()
  f.readline() #Number of grid points
  ngrid=int(f.readline().split()[0])

  r=[]
  f.readline()
  for i in range(ngrid):
    r.append(float(f.readline()))
  
  pot=[]
  while 'potential' in f.readline():
    tmp=[]
    for i in range(ngrid):
      tmp.append(float(f.readline()))
    pot.append(tmp)

  return effc,r,pot

####################################################
def read_trail_wf(f):
  ang_mom=[]
  r=[]
  wfs=[]
  label=f.readline().split()[1]
  f.readline()
  f.readline()
  f.readline()
  norb=int(f.readline())
  while 'Radial' not in f.readline():
    pass
  ngrid=int(f.readline())
  for i in range(ngrid):
    r.append(float(f.readline()))
  for o in range(norb):
    f.readline()
    ang_mom.append(int(f.readline().split()[2]))
    tmp=[]
    for i in range(ngrid):
      tmp.append(float(f.readline()))
    wfs.append(tmp)
  return label,ang_mom,r,wfs



####################################################
import numpy as np
import scipy
import scipy.interpolate
import matplotlib.pyplot as plt

def write_qwalk_pot(f,effc,r,pot,label="XX"):
  local=np.array(pot[-1])
  r=np.array(r)
  f.write("PSEUDO { %s \n"%label)
  f.write("AIP 12 \n")
  f.write("ADD_ZEFF \n")
  sub_pots=[]
  for v in pot[0:-1]:
    sub_pots.append(0.5*(np.array(v)-local)/r)
  sub_pots.append((0.5*local)/r)

  ngrid=len(r)
  f.write("BASIS { \n %s \n AOSPLINE \n NORENORMALIZE \n"%label)
  nregrid=400
  xgrid=np.linspace(0,4.0,nregrid)
  plt.subplots(1,1)
  for v in sub_pots:
    v[0]=v[1]
    vgrid=scipy.interpolate.griddata(r,v,xgrid,method='cubic')
    plt.plot(r,v,label='orig',lw=1)
    plt.plot(xgrid,vgrid,label='regrid',lw=1)
    
    f.write("SPLINE { \n S \n")
    for i in range(nregrid):
      f.write("%.9f %.9f \n"%(xgrid[i],vgrid[i]))
    f.write("}\n")
  f.write("} \n } \n")
  plt.xlim(0,2)
  plt.legend()
  plt.savefig("ecp_regrid.pdf",bbox_inches='tight')
  
    

####################################################

def write_qwalk_basis(f,ang_mom,r,wfs,label="XX"):
  translate={0:'S',1:'P',2:'5D',3:'7F'}
  r=np.array(r)
  ngrid=len(r)
  f.write("BASIS { \n %s \n AOSPLINE \n NORENORMALIZE \n"%label)
  nregrid=400
  xgrid=np.linspace(0,20.0,nregrid)
  plt.subplots(1,1)
  for l,v in zip(ang_mom,wfs):
    v=np.array(v)
    v=v/(r**(l+1))
    v[0]=v[1]
    vgrid=scipy.interpolate.griddata(r,v,xgrid,method='linear')

    plt.plot(r,v,label='orig'+str(translate[l]),lw=1)
    plt.plot(xgrid,vgrid,label='regrid'+str(translate[l]),lw=1)
    f.write("SPLINE { \n  %s \n"%translate[l])
    for i in range(nregrid):
      f.write("%.9f %.9f \n"%(xgrid[i],vgrid[i]))
    f.write("}\n")
  f.write("} \n ")
  plt.xlim(0,4)
  plt.legend()
  plt.savefig("basis_regrid.pdf",bbox_inches='tight')
####################################################
 
from scipy import optimize

def coeff_normalization(exp,angmom):

  fac=np.sqrt(2.*exp/np.pi)
  feg=4.*exp
  
  if angmom=="S" or angmom=='s' or angmom==0:
    return np.sqrt(2.*feg*fac)
  elif angmom=="P" or angmom=='p' or angmom==1:
    return np.sqrt(2.*feg*feg*fac/3.)
  elif angmom=="5D" or angmom=='d' or angmom==2:
    return np.sqrt(2.*feg*feg*feg*fac/15.)
####################################################

def sum_gauss(p,x):
  ng=int(len(p)/2)
  x=np.array(x)
  f=np.zeros(x.shape)
  for i in range(ng):
    f+=p[i]*np.exp(-p[ng+i]*x**2)
  return f
####################################################

def fit_gaussian_basis(x,y,angmom):
  nregrid=400
  xfit=np.linspace(1e-5,12.0,nregrid)
  yfit=scipy.interpolate.griddata(np.array(x),np.array(y),xfit,method='cubic')
  
  yfit=np.array(yfit)/np.array(xfit)**(angmom+1)
  
  
  errfunc=lambda p,x,y: sum_gauss(p,x)-y 
  objfunc=lambda p,x,y: np.sum(errfunc(p,x,y)**2/len(x))
  for ng in range(4,10):
    p0=[]
    for i in range(ng):
      p0.append(0.2)
    for i in range(ng):
      p0.append(0.5*2**i)
    #p1,cov,infodict,mesg,success=optimize.leastsq(errfunc,p0[:],args=(xfit,yfit),full_output=True)
    optres=optimize.basinhopping(objfunc,p0[:],
                                 disp=True,
                                 minimizer_kwargs={
                                    'args':(xfit,yfit),
                                    'options':{'disp':True}  },
                                    stepsize=2.0
                                )
    p1=optres.x
    coeffnorm=coeff_normalization(np.array(p1[ng:]),angmom)
    print("angular momentum",angmom)
    for i in range(ng):
      print(i+1,p1[i+ng],p1[i]/coeffnorm[i])
    #print('rms',np.sqrt(np.sum(infodict['fvec']**2)/len(infodict['fvec'])),'ng',ng)
    print('rms',np.sqrt(optres.fun))




####################################################

def compare_to_gaussian(f,r,wfs):
  while 'GAMESS' not in f.readline():
    pass


  mom_exp={"S":1,"P":2,"5D":3}
  nstate=len(wfs)
  fig,axes=plt.subplots(nstate,1,figsize=(4,8))
  #axes=[axes]
  count=0
  while True:
    line=f.readline()
    if '}' in line:
      break
    spl=line.split()
    angmom=spl[0]
    ngauss=int(spl[1])

    coeff=np.zeros(ngauss)
    exp=np.zeros(ngauss)
    for i in range(ngauss):
      spl=f.readline().split()
      coeff[i]=float(spl[2])
      exp[i]=float(spl[1])
      
    coeff=coeff*coeff_normalization(exp,angmom)
    print(coeff,exp)
    ygauss=np.zeros(len(r))
    for i in range(ngauss):
      ygauss+=coeff[i]*np.exp(-exp[i]*np.array(r)**2)
    axes[count].plot(r,ygauss,label='gaussian',lw=1)
    axes[count].plot(r,np.array(wfs[count])/np.array(r)**mom_exp[angmom],label='tabulated',lw=1)
    axes[count].set_xlim(0,9.0)
    axes[count].legend()
    count+=1
  plt.savefig("compare_to_gauss.pdf",bbox_inches='tight')




####################################################


if __name__=="__main__":
  effc,r,pot=read_trail_ecp(open("pp.data"))
  label,ang_mom,rbasis,wfbasis=read_trail_wf(open("awfn.data_d6s2_5D.txt"))
  compare_to_gaussian(open("gaussbasis"),rbasis,wfbasis)
  #fit_gaussian_basis(rbasis,wfbasis[0],ang_mom[0])  
  #fit_gaussian_basis(rbasis,wfbasis[1],ang_mom[1])
  #write_qwalk_pot(open("Fe.ecp",'w'),effc,r,pot,label)
  #write_qwalk_basis(open("Fe.basis",'w'),ang_mom,rbasis,wfbasis,label=label)

  
